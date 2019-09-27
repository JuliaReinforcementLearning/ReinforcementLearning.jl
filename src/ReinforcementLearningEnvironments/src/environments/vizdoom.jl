using ViZDoom
const vz = ViZDoom

export ViZDoomEnv, basic_ViZDoom_env

#####
##### helper functions
#####

function listconsts(typ)
    allconsts = names(vz, all = true)
    idx = findall(x -> typeof(getfield(vz, x)) == getfield(vz, typ), allconsts)
    allconsts[idx]
end

list_available_buttons() = listconsts(:Button)
list_screen_resolution() = listconsts(:ScreenResolution)
list_screen_format() = listconsts(:ScreenFormat)
list_mode() = listconsts(:Mode)

function list_options()
    allconsts = names(vz, all = true)
    idx = findall(x -> length(string(x)) > 2 && string(x)[1:3] == "set", allconsts)
    map(x -> Symbol(string(x)[5:end]), allconsts[idx])
end

struct ViZDoomEnv{O,A} <: AbstractEnv
    game::vz.DoomGameAllocated
    actions::Array{Array{Float64,1},1}
    sleeptime::Float64
    observation_space::O
    action_space::A
end

observation_space(env::ViZDoomEnv) = env.observation_space
action_space(env::ViZDoomEnv) = env.action_space

function basic_ViZDoom_env(; add_game_args = "", kw...)
    defaults = (
        screen_format = :GRAY8,
        screen_resolution = :RES_160X120,
        window_visible = false,
        living_reward = 0,
        episode_timeout = 500,
    )
    config = Dict(pairs(merge(defaults, kw)))
    for (k, v) in config
        if typeof(v) == Symbol
            config[k] = getfield(vz, v)
        elseif typeof(v) <: AbstractArray && typeof(v[1]) == Symbol
            config[k] = map(x -> getfield(vz, x), v)
        end
    end
    game = vz.basic_game(; config...)
    vz.add_game_args(game, add_game_args)
    if config[:window_visible]
        sleeptime = 1.0 / vz.DEFAULT_TICRATE
    else
        sleeptime = 0.
    end
    na = haskey(config, :available_buttons) ? length(config[:available_buttons]) : 3
    actions = [Float64[i == j for i = 1:na] for j = 1:na]
    screen_size = vz.get_screen_size(game)
    env = ViZDoomEnv(
        game,
        actions,
        sleeptime,
        MultiDiscreteSpace(
            fill(typemax(UInt8), screen_size),
            fill(typemin(UInt8), screen_size),
        ),
        DiscreteSpace(na),
    )
    vz.init(env.game)
    env
end

function interact!(env::ViZDoomEnv, a)
    r = vz.make_action(env.game, env.actions[a])
    done = vz.is_episode_finished(env.game)
    if done
        state = zeros(UInt8, vz.get_screen_size(env.game))
    else
        state = vz.get_screen_buffer(env.game)
    end
    if env.sleeptime > 0
        sleep(env.sleeptime)
    end
    (observation = state, reward = r, isdone = done)
end

function reset!(env::ViZDoomEnv)
    vz.new_episode(env.game)
    vz.get_screen_buffer(env.game)
    nothing
end

function observe(env::ViZDoomEnv)
    (
     observation = vz.get_screen_buffer(env.game),
     isdone = vz.is_episode_finished(env.game),
    )
end