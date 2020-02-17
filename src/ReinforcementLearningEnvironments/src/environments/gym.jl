using PyCall

export GymEnv

# TODO: support `seed`

struct GymEnv{T,Ta<:AbstractSpace,To<:AbstractSpace} <: AbstractEnv
    pyenv::PyObject
    observation_space::To
    action_space::Ta
    state::PyObject
end

function GymEnv(name::String)
    gym = pyimport("gym")
    pyenv = gym.make(name)
    obs_space = convert(AbstractSpace, pyenv.observation_space)
    act_space = convert(AbstractSpace, pyenv.action_space)
    obs_type =
        if obs_space isa Union{MultiContinuousSpace,MultiDiscreteSpace}
            PyArray
        elseif obs_space isa ContinuousSpace
            Float64
        elseif obs_space isa DiscreteSpace
            Int
        elseif obs_space isa TupleSpace
            PyVector
        elseif obs_space isa DictSpace
            PyDict
        else
            error("don't know how to get the observation type from observation space of $obs_space")
        end
    env = GymEnv{obs_type,typeof(act_space),typeof(obs_space)}(
        pyenv,
        obs_space,
        act_space,
        PyNULL(),
    )
    reset!(env) # reset immediately to init env.state
    env
end

function (env::GymEnv{T})(action) where {T}
    pycall!(env.state, env.pyenv.step, PyObject, action)
    nothing
end

function RLBase.reset!(env::GymEnv)
    pycall!(env.state, env.pyenv.reset, PyObject)
    nothing
end

function RLBase.observe(env::GymEnv{T}) where {T}
    if pyisinstance(env.state, PyCall.@pyglobalobj :PyTuple_Type)
        obs, reward, isdone, info = convert(Tuple{T,Float64,Bool,PyDict}, env.state)
        (reward = reward, terminal = isdone, state = obs)
    else
        # env has just been reseted
        (
            reward = 0.,  # dummy
            terminal = false,
            state = convert(T, env.state),
        )
    end
end

RLBase.render(env::GymEnv) = env.pyenv.render()

###
### utils
###

function Base.convert(::Type{AbstractSpace}, s::PyObject)
    spacetype = s.__class__.__name__
    if spacetype == "Box"
        MultiContinuousSpace(s.low, s.high)
    elseif spacetype == "Discrete"  # for GymEnv("CliffWalking-v0"), `s.n` is of type PyObject (numpy.int64)
        DiscreteSpace(0, py"int($s.n)" - 1)  # for GymEnv("CliffWalking-v0"), `s.n` is of type PyObject (numpy.int64)
    elseif spacetype == "MultiBinary"
        MultiDiscreteSpace(zeros(Int8, s.n), ones(Int8, s.n))
    elseif spacetype == "MultiDiscrete"
        MultiDiscreteSpace(
            zeros(eltype(s.nvec), size(s.nvec)),
            s.nvec .- one(eltype(s.nvec)),
        )
    elseif spacetype == "Tuple"
        TupleSpace((convert(AbstractSpace, x) for x in s.spaces)...)
    elseif spacetype == "Dict"
        DictSpace((k => convert(AbstractSpace, v) for (k, v) in s.spaces)...)
    else
        error("Don't know how to convert Gym Space of class [$(spacetype)]")
    end
end

function list_gym_env_names(;
    modules = [
        "gym.envs.algorithmic",
        "gym.envs.box2d",
        "gym.envs.classic_control",
        "gym.envs.mujoco",
        "gym.envs.mujoco.ant_v3",
        "gym.envs.mujoco.half_cheetah_v3",
        "gym.envs.mujoco.hopper_v3",
        "gym.envs.mujoco.humanoid_v3",
        "gym.envs.mujoco.swimmer_v3",
        "gym.envs.mujoco.walker2d_v3",
        "gym.envs.robotics",
        "gym.envs.toy_text",
        "gym.envs.unittest",
    ])
    gym = pyimport("gym")
    [x.id for x in gym.envs.registry.all() if split(x.entry_point, ':')[1] in modules]
end