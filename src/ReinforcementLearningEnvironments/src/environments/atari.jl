using ArcadeLearningEnvironment, GR, Random

export AtariEnv

mutable struct AtariEnv{IsGrayScale, TerminalOnLifeLoss, N, S<:AbstractRNG} <: AbstractEnv
    ale::Ptr{Nothing}
    screens::Tuple{Array{UInt8, N}, Array{UInt8, N}}  # for max-pooling
    actions::Vector{Int64}
    action_space::DiscreteSpace{UnitRange{Int}}
    observation_space::MultiDiscreteSpace{Array{UInt8, N}}
    noopmax::Int
    frame_skip::Int
    reward::Float32
    lives::Int
    seed::S
end

"""
    AtariEnv(;kwargs...)

This implementation follows the guidelines in [Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents](https://arxiv.org/abs/1709.06009)

TODO: support seed! in single/multi thread

# Keywords

- `name::String="pong"`: name of the Atari environments. Use `getROMList` to show all supported environments.
- `grayscale_obs::Bool=true`:if `true`, then gray scale observation is returned, otherwise, RGB observation is returned.
- `noop_max::Int=30`: max number of no-ops.
- `frame_skip::Int=4`: the frequency at which the agent experiences the game.
- `terminal_on_life_loss::Bool=false`: if `true`, then game is over whenever a life is lost.
- `repeat_action_probability::Float64=0.`
- `color_averaging::Bool=false`: whether to perform phosphor averaging or not.
- `max_num_frames_per_episode::Int=0`
- `full_action_space::Bool=false`: by default, only use minimal action set. If `true`, one need to call `legal_actions` to get the valid action set. TODO

See also the [python implementation](https://github.com/openai/gym/blob/c072172d64bdcd74313d97395436c592dc836d5c/gym/wrappers/atari_preprocessing.py#L8-L36)
"""
function AtariEnv(
    ;name = "pong",
    grayscale_obs=true,
    noop_max = 30,
    frame_skip = 4,
    terminal_on_life_loss=false,
    repeat_action_probability=0.,
    color_averaging=false,
    max_num_frames_per_episode=0,
    full_action_space=false,
    seed=nothing
)
    frame_skip > 0 || throw(ArgumentError("frame_skip must be greater than 0!"))
    name in getROMList() || throw(ArgumentError("unknown ROM name! run `getROMList()` to see all the game names."))

    if isnothing(seed)
        seed = (MersenneTwister(), 0)
    elseif seed isa Tuple{Int, Int}
        seed = (MersenneTwister(seed[1]), seed[2])
    else
        @error "You must specify two seeds, one for Julia wrapper, one for internal C implementation" # ??? maybe auto generate two seed from one
    end

    ale = ALE_new()
    setInt(ale, "random_seed", seed[2])
    setInt(ale, "frame_skip", Int32(1))  # !!! do not use internal frame_skip here, we need to apply max-pooling for the latest two frames, so we need to manually implement the mechanism.
    setInt(ale, "max_num_frames_per_episode", max_num_frames_per_episode)
    setFloat(ale, "repeat_action_probability", Float32(repeat_action_probability))
    setBool(ale, "color_averaging", color_averaging)
    loadROM(ale, name)

    observation_size = grayscale_obs ? (getScreenWidth(ale), getScreenHeight(ale)) : (3, getScreenWidth(ale), getScreenHeight(ale))  # !!! note the order
    observation_space = MultiDiscreteSpace(
        fill(typemin(Cuchar), observation_size),
        fill(typemax(Cuchar), observation_size),
    )

    actions = full_action_space ? getLegalActionSet(ale) : getMinimalActionSet(ale)
    action_space = DiscreteSpace(length(actions))
    screens = (
        fill(typemin(Cuchar), observation_size),
        fill(typemin(Cuchar), observation_size),
    )

    AtariEnv{grayscale_obs, terminal_on_life_loss, grayscale_obs ? 2 : 3, typeof(seed[1])}(
        ale,
        screens,
        actions,
        action_space,
        observation_space,
        noop_max,
        frame_skip,
        0.0f0,
        lives(ale),
        seed[1]
    )
end

update_screen!(env::AtariEnv{true}, screen) = ArcadeLearningEnvironment.getScreenGrayscale!(env.ale, vec(screen))
update_screen!(env::AtariEnv{false}, screen) = ArcadeLearningEnvironment.getScreenRGB!(env.ale, vec(screen))

function (env::AtariEnv{is_gray_scale, is_terminal_on_life_loss})(a) where {is_gray_scale, is_terminal_on_life_loss}
    r = 0.0f0

    for i in 1:env.frame_skip
        r += act(env.ale, env.actions[a])
        if i == env.frame_skip
            update_screen!(env, env.screens[1])
        elseif i == env.frame_skip - 1
            update_screen!(env, env.screens[2])
        end
    end

    # max-pooling
    if env.frame_skip > 1
        env.screens[1] .= max.(env.screens[1], env.screens[2])
    end

    env.reward = r
    nothing
end

is_terminal(env::AtariEnv{<:Any, true}) = game_over(env.ale) || (lives(env.ale) < env.lives)
is_terminal(env::AtariEnv{<:Any, false}) = game_over(env.ale)

RLBase.observe(env::AtariEnv) = (reward = env.reward, terminal = is_terminal(env), state = env.screens[1])

function RLBase.reset!(env::AtariEnv)
    reset_game(env.ale)
    for _ = 1:rand(env.seed, 0:env.noopmax)
        act(env.ale, Int32(0))
    end
    update_screen!(env, env.screens[1])  # no need to update env.screens[2]
    env.reward = 0.0f0  # dummy
    env.lives = lives(env.ale)
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

function RLBase.render(env::AtariEnv)
    x = getScreenRGB(env.ale)
    imshowcolor(x, (Int(getScreenWidth(env.ale)), Int(getScreenHeight(env.ale))))
end

list_atari_rom_names() = getROMList()
