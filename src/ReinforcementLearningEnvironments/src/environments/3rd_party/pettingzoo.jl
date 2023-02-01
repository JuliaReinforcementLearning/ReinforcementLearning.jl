using .PyCall


np = pyimport("numpy")

export PettingzooEnv

"""
    PettingzooEnv(;kwargs...)

`PettingzooEnv` is an interface of the python library pettingzoo for multi agent reinforcement learning environments. It can be used to test multi
    agent reinforcement learning algorithms implemented in JUlia ReinforcementLearning.
"""
function PettingzooEnv(name::String; seed=123, args...)
    if !PyCall.pyexists("pettingzoo.$name")
       error("Cannot import pettingzoo.$name")
    end
    pz = pyimport("pettingzoo.$name")
    pyenv = pz.env(;args...)
    pyenv.reset(seed=seed)
    obs_space = space_transform(pyenv.observation_space(pyenv.agents[1]))
    act_space = space_transform(pyenv.action_space(pyenv.agents[1]))
    env = PettingzooEnv{typeof(act_space),typeof(obs_space),typeof(pyenv)}(
        pyenv,
        obs_space,
        act_space,
        PyNULL,
        seed,
        1
    )
    env
end

# basic function needed for simulation ========================================================================

function RLBase.reset!(env::PettingzooEnv)
    pycall!(env.state, env.pyenv.reset, PyObject, env.seed)
    env.ts = 1
    nothing
end

function RLBase.is_terminated(env::PettingzooEnv)
    _, _, t, d, _ = pycall(env.pyenv.last, PyObject)
    t || d
end



## State / observation implementations ========================================================================

RLBase.state(env::PettingzooEnv, ::Observation{Any}, players::Tuple) = Dict(p => state(env, p) for p in players)


# partial obsverability is default for pettingzoo
function RLBase.state(env::PettingzooEnv, ::Observation{Any}, player)
    env.pyenv.observe(player)
end


## state space =========================================================================================================================================

RLBase.state_space(env::PettingzooEnv, ::Observation{Any}, players) = Space(Dict(player => state_space(env, player) for player in players))

    # partial observability
RLBase.state_space(env::PettingzooEnv, ::Observation{Any}, player::String) = space_transform(env.pyenv.observation_space(player))

# for full observability. Be carefule: action_space has also to be adjusted
# RLBase.state_space(env::PettingzooEnv, ::Observation{Any}, player::String) = space_transform(env.pyenv.state_space)


## action space implementations ====================================================================================

RLBase.action_space(env::PettingzooEnv, players::Tuple{String}) =
         Space(Dict(p => action_space(env, p) for p in players))

RLBase.action_space(env::PettingzooEnv, player::String) = space_transform(env.pyenv.action_space(player))

RLBase.action_space(env::PettingzooEnv, player::Integer) = space_transform(env.pyenv.action_space(env.pyenv.agents[player]))

RLBase.action_space(env::PettingzooEnv, player::DefaultPlayer) = env.action_space

## action functions ========================================================================================================================

function (env::PettingzooEnv)(actions::Dict, players::Tuple)
    @assert length(actions) == length(players)
    env.ts += 1
    for p in players
        env(actions[p])
    end
end

function (env::PettingzooEnv)(actions::Dict, player)
    @assert length(actions) == length(players(env))
    for p in players(env)
        env(actions[p])
    end
end

function (env::PettingzooEnv)(actions::Dict{String, Int})
    @assert length(actions) == length(players(env))
    for p in env.pyenv.agents
        pycall(env.pyenv.step, PyObject, actions[p])
    end
end

function (env::PettingzooEnv)(actions::Dict{String, Real})
    @assert length(actions) == length(players(env))
    env.ts += 1
    for p in env.pyenv.agents
        pycall(env.pyenv.step, PyObject, np.array(actions[p]; dtype=np.float32))
    end
end

function (env::PettingzooEnv)(action::Vector)
    pycall(env.pyenv.step, PyObject, np.array(action; dtype=np.float32))
end

function (env::PettingzooEnv)(action::Integer)
    env.ts += 1
    pycall(env.pyenv.step, PyObject, action)
end

# reward of player ======================================================================================================================
function RLBase.reward(env::PettingzooEnv, player::String)
    env.pyenv.rewards[player]
end


# Multi agent part =========================================================================================================================================


RLBase.players(env::PettingzooEnv) = env.pyenv.agents

function RLBase.current_player(env::PettingzooEnv, post_action=false)
    cur_id = env.ts % length(env.pyenv.agents) == 0 ? length(env.pyenv.agents) : env.ts % length(env.pyenv.agents)
    cur_id = post_action ? (cur_id - 1 == 0 ? length(env.pyenv.agents) : cur_id - 1) : cur_id
    return env.pyenv.agents[cur_id]
end

function RLBase.NumAgentStyle(env::PettingzooEnv)
    n = length(env.pyenv.agents)
    if n == 1
        SingleAgent()
    else
        MultiAgent(n)
    end
end


RLBase.DynamicStyle(::PettingzooEnv) = SEQUENTIAL
RLBase.ActionStyle(::PettingzooEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::PettingzooEnv) = IMPERFECT_INFORMATION
RLBase.ChanceStyle(::PettingzooEnv) = EXPLICIT_STOCHASTIC

