import Images: imresize
using ArcadeLearningEnvironment, Parameters

struct AtariEnv
    ale::Ptr{Void}
    screen::Array{UInt8, 1}
    getscreen::Function
    noopmax::Int64
end
function AtariEnv(name; 
                  colorspace = "Grayscale",
                  frame_skip = 4, noopmax = 20,
                  color_averaging = true,
                  repeat_action_probability = 0.,
                  romdir = joinpath(@__DIR__, "atariroms"))
    if !isdir(romdir) getroms(romdir) end
    path = joinpath(romdir, name * ".bin")
    if isfile(path)
        ale = ALE_new()
        loadROM(ale, path)
        setBool(ale, "color_averaging", color_averaging)
        setInt(ale, "frame_skip", Int32(frame_skip))
        setFloat(ale, "repeat_action_probability", 
                 Float32(repeat_action_probability))
    else
        error("ROM $path not found.")
    end
    if colorspace == "Grayscale"
        screen = Array{Cuchar}(210*160)
        getscreen = getScreenGrayscale
    elseif colorspace == "RGB"
        screen = Array{Cuchar}(3*210*160)
        getscreen = getScreenRGB
    elseif colorspace == "Raw"
        screen = Array{Cuchar}(210*160)
        getscreen = getScreen
    end
    AtariEnv(ale, screen, getscreen, noopmax)
end

import ArcadeLearningEnvironment.getScreen
function getScreen(p::Ptr, s::Array{Cuchar, 1})
    sraw = getScreen(p)
    for i in 1:length(s)
        s[i] =  sraw[i] .>> 1
    end
end

function getroms(romdir)
    info("Downloading roms to $romdir")
    tmpdir = mktempdir()
    Base.LibGit2.clone("https://github.com/openai/atari-py", tmpdir)
    mv(joinpath(tmpdir, "atari_py", "atari_roms"), romdir)
    rm(tmpdir, recursive = true, force = true)
end
listroms(romdir = joinpath(@__DIR__, "atariroms")) = readdir(romdir)

import ReinforcementLearning: interact!, getstate, reset!, 
preprocessstate, selectaction, callback!

function interact!(a, env::AtariEnv)
    r = act(env.ale, Int32(a - 1))
    env.getscreen(env.ale, env.screen)
    env.screen, r, game_over(env.ale)
end
function getstate(env::AtariEnv)
    env.getscreen(env.ale, env.screen)
    env.screen, game_over(env.ale)
end
function reset!(env::AtariEnv)
    reset_game(env.ale)
    for _ in 1:rand(0:env.noopmax) act(env.ale, Int32(0)) end
    nothing
end

@with_kw struct AtariPreprocessor
    gpu::Bool = false
    croptosquare::Bool = false
    cropfromrow::Int64 = 34
    dimx::Int64 = 80
    dimy::Int64 = croptosquare ? 80 : 105
    scale::Bool = false
    inputtype::DataType = scale ? Float32 : UInt8
end
togpu(x) = CuArrays.adapt(CuArray, x)
function preprocessstate(p::AtariPreprocessor, s)
    if p.croptosquare
        tmp = reshape(s, 160, 210)[:,p.cropfromrow:p.cropfromrow + 159]
        small = reshape(imresize(tmp, p.dimx, p.dimy), p.dimx, p.dimy, 1)
    else
        small = reshape(imresize(reshape(s, 160, 210), p.dimx, p.dimy), 
                        p.dimx, p.dimy, 1)
    end
    if p.scale
        scale!(small, 1/255)
    else
        small = ceil.(p.inputtype, small)
    end
    if p.gpu
        togpu(small)
    else
        p.inputtype.(small)
    end
end
function preprocessstate(p::AtariPreprocessor, ::Void)
    s = zeros(p.inputtype, p.dimx, p.dimy, 1)
    if p.gpu
        togpu(s)
    else
        s
    end
end
