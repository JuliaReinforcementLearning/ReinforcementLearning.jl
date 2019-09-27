using ArcadeLearningEnvironment, GR

export AtariEnv

mutable struct AtariEnv{To,F} <: AbstractEnv
    ale::Ptr{Nothing}
    screen::Array{UInt8,1}
    getscreen!::F
    actions::Array{Int64,1}
    action_space::DiscreteSpace{Int}
    observation_space::To
    noopmax::Int
    reward::Float32
end

"""
    AtariEnv(name; colorspace = "Grayscale", frame_skip = 4, noopmax = 20,
                   color_averaging = true, repeat_action_probability = 0.)
Returns an AtariEnv that can be used in an RLSetup of the
[ReinforcementLearning](https://github.com/jbrea/ReinforcementLearning.jl)
package. Check the deps/roms folder of the ArcadeLearningEnvironment package to
see all available `name`s.
"""
function AtariEnv(
    name;
    colorspace = "Grayscale",
    frame_skip = 4,
    noopmax = 20,
    color_averaging = true,
    actionset = :minimal,
    repeat_action_probability = 0.,
)
    ale = ALE_new()
    setBool(ale, "color_averaging", color_averaging)
    setInt(ale, "frame_skip", Int32(frame_skip))
    setFloat(ale, "repeat_action_probability", Float32(repeat_action_probability))
    loadROM(ale, name)
    observation_length = getScreenWidth(ale) * getScreenHeight(ale)
    if colorspace == "Grayscale"
        screen = Array{Cuchar}(undef, observation_length)
        getscreen! = ArcadeLearningEnvironment.getScreenGrayscale!
        observation_space = MultiDiscreteSpace(
            fill(typemax(Cuchar), observation_length),
            fill(typemin(Cuchar), observation_length),
        )
    elseif colorspace == "RGB"
        screen = Array{Cuchar}(undef, 3 * observation_length)
        getscreen! = ArcadeLearningEnvironment.getScreenRGB!
        observation_space = MultiDiscreteSpace(
            fill(typemax(Cuchar), 3 * observation_length),
            fill(typemin(Cuchar), 3 * observation_length),
        )
    elseif colorspace == "Raw"
        screen = Array{Cuchar}(undef, observation_length)
        getscreen! = ArcadeLearningEnvironment.getScreen!
        observation_space = MultiDiscreteSpace(
            fill(typemax(Cuchar), observation_length),
            fill(typemin(Cuchar), observation_length),
        )
    end
    actions = actionset == :minimal ? getMinimalActionSet(ale) : getLegalActionSet(ale)
    action_space = DiscreteSpace(length(actions))
    AtariEnv(
        ale,
        screen,
        getscreen!,
        actions,
        action_space,
        observation_space,
        noopmax,
        0.0f0,
    )
end

function interact!(env::AtariEnv, a)
    env.reward = act(env.ale, env.actions[a])
    env.getscreen!(env.ale, env.screen)
    nothing
end

observe(env::AtariEnv) =
    Observation(reward = env.reward, terminal = game_over(env.ale), state = env.screen)

function reset!(env::AtariEnv)
    reset_game(env.ale)
    for _ = 1:rand(0:env.noopmax)
        act(env.ale, Int32(0))
    end
    env.getscreen!(env.ale, env.screen)
    env.reward = 0.0f0  # dummy
    nothing
end


imshowgrey(x::Array{UInt8,2}) = imshowgrey(x[:], size(x))
imshowgrey(x::Array{UInt8,1}, dims) = imshow(reshape(x, dims), colormap = 2)
imshowcolor(x::Array{UInt8,3}) = imshowcolor(x[:], size(x))

function imshowcolor(x::Array{UInt8,1}, dims)
    clearws()
    setviewport(0, dims[1] / dims[2], 0, 1)
    setwindow(0, 1, 0, 1)
    y = (zeros(UInt32, dims...) .+ 0xff) .<< 24
    img = UInt32.(x)
    @simd for i = 1:length(y)
        @inbounds y[i] += img[3*(i-1)+1] + img[3*(i-1)+2] << 8 + img[3*i] << 16
    end
    drawimage(0, 1, 0, 1, dims..., y)
    updatews()
end

function render(env::AtariEnv)
    x = getScreenRGB(env.ale)
    imshowcolor(x, (Int(getScreenWidth(env.ale)), Int(getScreenHeight(env.ale))))
end

list_atari_rom_names() = getROMList()
