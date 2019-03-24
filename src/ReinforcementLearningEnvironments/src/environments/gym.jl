using PyCall

export GymEnv

# TODO: support `seed`

struct GymEnv{Ta<:AbstractSpace, To<:AbstractSpace} <: AbstractEnv
    pyenv::PyObject
    observation_space::To
    action_space::Ta
    state::PyObject
end

function GymEnv(name::String)
    gym = pyimport("gym")
    pyenv = gym.make(name)
    obsspace = gymspace2jlspace(pyenv.observation_space)
    actspace = gymspace2jlspace(pyenv.action_space)
    state = PyNULL()
    env = GymEnv(pyenv, obsspace, actspace, state)
    reset!(env) # reset!
    env
end

action_space(env::GymEnv) = env.action_space
observation_space(env::GymEnv) = env.observation_space

function interact!(env::GymEnv, action)
    pycall!(env.state, env.pyenv.step, PyObject, action)
    obs, reward, isdone, info = convert(Tuple{PyArray, Float64, Bool, PyDict}, env.state)
    (observation=obs, reward=reward, isdone=isdone)
end

function reset!(env::GymEnv)
    pycall!(env.state, env.pyenv.reset, PyObject)
    nothing
end

function observe(env::GymEnv) 
    if pyisinstance(env.state, PyCall.@pyglobalobj :PyTuple_Type)
        obs, reward, isdone, info = convert(Tuple{PyArray, Float64, Bool, PyDict}, env.state)
        (observation=obs, isdone=isdone)
    else
        # env has just been reseted
        (observation=PyArray(env.state), isdone=false)
    end
end

render(env::GymEnv) = env.pyenv.render()

###
### utils
###

function gymspace2jlspace(s::PyObject)
    spacetype = s.__class__.__name__
    if     spacetype == "Box"           MultiContinuousSpace(s.low, s.high)
    elseif spacetype == "Discrete"      DiscreteSpace(py"int($s.n)" - 1, 0)  # for GymEnv("CliffWalking-v0"), `s.n` is of type PyObject (numpy.int64)
    elseif spacetype == "MultiBinary"   MultiDiscreteSpace(ones(Int8, s.n), zeros(Int8, s.n))
    elseif spacetype == "MultiDiscrete" MultiDiscreteSpace(s.nvec .- one(eltype(s.nvec)), zeros(eltype(s.nvec), size(s.nvec)))
    elseif spacetype == "Tuple"         TupleSpace((gymspace2jlspace(x) for x in s.spaces)...)
    elseif spacetype == "Dict"          DictSpace((k => gymspace2jlspace(v) for (k, v) in s.spaces)...)
    else error("Don't know how to convert Gym Space of class [$(spacetype)]")
    end
end

function list_gym_env_names(;
    modules=[
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
        "gym.envs.unittest"])
    gym = pyimport("gym")
    [x.id for x in gym.envs.registry.all() if split(x._entry_point, ':')[1] in modules]
end