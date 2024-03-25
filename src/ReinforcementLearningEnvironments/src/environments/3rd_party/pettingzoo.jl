using .PyCall

np = pyimport("numpy")

export PettingZooEnv


"""
    PettingZooEnv(;kwargs...)

`PettingZooEnv` is an interface of the python library Pettingzoo for multi agent reinforcement learning environments. It can be used to test multi
    agent reinforcement learning algorithms implemented in JUlia ReinforcementLearning.
"""

function PettingZooEnv(name::String; seed=123, args...)
    if !PyCall.pyexists("pettingzoo.$name")
       error("Cannot import pettingzoo.$name")
    end
    pz = pyimport("pettingzoo.$name")
    pyenv = pz.env(;args...)
    pyenv.reset(seed=seed)
    obs_space = space_transform(pyenv.observation_space(pyenv.agents[1]))
    act_space = space_transform(pyenv.action_space(pyenv.agents[1]))
    env = PettingZooEnv{typeof(act_space),typeof(obs_space),typeof(pyenv)}(
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

function RLBase.reset!(env::PettingZooEnv)
    pycall!(env.state, env.pyenv.reset, PyObject, env.seed)
    nothing
end

function RLBase.is_terminated(env::PettingZooEnv)
    _, _, t, d, _ = pycall(env.pyenv.last, PyObject)
    t || d
end



## State / observation implementations ========================================================================

RLBase.state(env::PettingZooEnv, ::Observation{Any}, players::Tuple) = Dict(p => state(env, p) for p in players)


# partial observability is default for pettingzoo
function RLBase.state(env::PettingZooEnv, ::Observation{Any}, player)
    env.pyenv.observe(player)
end


## state space =========================================================================================================================================

RLBase.state_space(env::PettingZooEnv, ::Observation{Any}, players) = Space(Dict(player => state_space(env, player) for player in players))

    # partial observability
RLBase.state_space(env::PettingZooEnv, ::Observation{Any}, player::Player) = space_transform(env.pyenv.observation_space(String(player)))

# for full observability. Be careful: action_space has also to be adjusted
# RLBase.state_space(env::PettingZooEnv, ::Observation{Any}, player::String) = space_transform(env.pyenv.state_space)


## action space implementations ====================================================================================

RLBase.action_space(env::PettingZooEnv, players::Tuple{Symbol}) =
         Space(Dict(p => action_space(env, p) for p in players))

RLBase.action_space(env::PettingZooEnv, player::Player) = space_transform(env.pyenv.action_space(String(player)))

RLBase.action_space(env::PettingZooEnv, player::Integer) = space_transform(env.pyenv.action_space(env.pyenv.agents[player]))

RLBase.action_space(env::PettingZooEnv, player::DefaultPlayer) = env.action_space

## action functions ========================================================================================================================

function RLBase.act!(env::PettingZooEnv, actions::Dict{Symbol, Int})
    @assert length(actions) == length(players(env))
    for p ∈ players(env)
        pycall(env.pyenv.step, PyObject, actions[p])
    end
end

function RLBase.act!(env::PettingZooEnv, actions::Dict{Symbol, Real})
    @assert length(actions) == length(env.pyenv.agents)
    for p ∈ players(env)
        pycall(env.pyenv.step, PyObject, np.array(actions[p]; dtype=np.float32))
    end
end

function RLBase.act!(env::PettingZooEnv, actions::Dict{Symbol, Vector})
    @assert length(actions) == length(env.pyenv.agents)
    for p ∈ players(env)
        RLBase.act!(env, p)
    end
end

function RLBase.act!(env::PettingZooEnv, actions::NamedTuple)
    @assert length(actions) == length(env.pyenv.agents)
    for player ∈ players(env)
        RLBase.act!(env, actions[player])
    end
end

# for vectors, pettingzoo need them to be in proper numpy type
function RLBase.act!(env::PettingZooEnv, action::Vector)
    pycall(env.pyenv.step, PyObject, np.array(action; dtype=np.float32))
end

function RLBase.act!(env::PettingZooEnv, action)
    pycall(env.pyenv.step, PyObject, action)
end

# reward of player ======================================================================================================================
function RLBase.reward(env::PettingZooEnv, player::Player)
    env.pyenv.rewards[String(player)]
end


# Multi agent part =========================================================================================================================================


RLBase.players(env::PettingZooEnv) = Symbol.(env.pyenv.agents)

function RLBase.current_player(env::PettingZooEnv)
    return Symbol(env.pyenv.agents[env.current_player])
end

function RLBase.next_player!(env::PettingZooEnv)
    env.current_player = env.current_player < length(env.pyenv.agents) ? env.current_player + 1 : 1
end

function RLBase.NumAgentStyle(env::PettingZooEnv)
    n = length(env.pyenv.agents)
    if n == 1
        SingleAgent()
    else
        MultiAgent(n)
    end
end

RLBase.DynamicStyle(::PettingZooEnv) = SIMULTANEOUS
RLBase.ActionStyle(::PettingZooEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::PettingZooEnv) = IMPERFECT_INFORMATION
RLBase.ChanceStyle(::PettingZooEnv) = EXPLICIT_STOCHASTIC

