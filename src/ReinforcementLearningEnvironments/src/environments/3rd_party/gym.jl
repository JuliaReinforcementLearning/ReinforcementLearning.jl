using .PyCall

function GymEnv(name::String; seed::Union{Int,Nothing}=nothing)
    if !PyCall.pyexists("gymnasium")
        error(
            "Cannot import module 'gymnasium'.\n\nIf you did not yet install it, try running\n`ReinforcementLearningEnvironments.install_gym()`\n",
        )
    end
    gym = pyimport_conda("gymnasium", "gymnasium")
    if PyCall.pyexists("d4rl")
        pyimport("d4rl")
    end
    pyenv = try
        gym.make(name)
    catch e
        error(
            "Gym environment $name not found.\n\nRun `ReinforcementLearningEnvironments.list_gym_env_names()` to find supported environments.\n",
        )
    end
    if seed !== nothing
        pyenv.seed(seed)
    end
    obs_space = space_transform(pyenv.observation_space)
    act_space = space_transform(pyenv.action_space)
    obs_type = if obs_space isa Space{<:Union{Array{<:Interval},Array{<:ZeroTo}}}
        PyArray
    elseif obs_space isa Interval
        Float64
    elseif obs_space isa ZeroTo
        Int
    elseif obs_space isa Space{<:Tuple}
        PyVector
    elseif obs_space isa Space{<:Dict}
        PyDict
    else
        error("don't know how to get the observation type from observation space of $obs_space")
    end
    env = GymEnv{obs_type,typeof(act_space),typeof(obs_space),typeof(pyenv)}(
        pyenv,
        obs_space,
        act_space,
        PyNULL(),
    )
    reset!(env) # reset immediately to init env.state
    env
end

Base.nameof(env::GymEnv) = env.pyenv.__class__.__name__

function Base.copy(env::GymEnv)
    @warn "clone method is not exposed in GymEnv"
    env
end

function RLBase.act!(env::GymEnv{T}, action) where {T}
    if env.action_space isa Tuple
        action = Tuple(action)
    end
    pycall!(env.state, env.pyenv.step, PyObject, action)
    nothing
end

function RLBase.reset!(env::GymEnv)
    pycall!(env.state, env.pyenv.reset, PyObject)
    nothing
end

RLBase.action_space(env::GymEnv) = env.action_space
RLBase.state_space(env::GymEnv) = env.observation_space

function RLBase.reward(env::GymEnv{T}) where {T}
    if pyisinstance(env.state, PyCall.@pyglobalobj :PyTuple_Type) && length(env.state) == 5
        _, reward, = convert(Tuple{T,Float64,Bool,Bool,PyDict}, env.state)
        reward
    elseif pyisinstance(env.state, PyCall.@pyglobalobj :PyTuple_Type) && length(env.state) == 4
        _, reward, = convert(Tuple{T,Float64,Bool,PyDict}, env.state)
        reward
    else
        0.0
    end
end

function RLBase.is_terminated(env::GymEnv{T}) where {T}
    if pyisinstance(env.state, PyCall.@pyglobalobj :PyTuple_Type) && length(env.state) == 5
        _, _, isterminated, istruncated, = convert(Tuple{T,Float64,Bool,Bool,PyDict}, env.state)
        isterminated || istruncated
    elseif pyisinstance(env.state, PyCall.@pyglobalobj :PyTuple_Type) && length(env.state) == 4
        @warn "Gym version outdated. Update gym to obtain termination and truncation info instead of done signal."
        _, _, isdone, = convert(Tuple{T,Float64,Bool,PyDict}, env.state)
        isdone
    else
        false
    end
end

function RLBase.state(env::GymEnv{T}) where {T}
    if pyisinstance(env.state, PyCall.@pyglobalobj :PyTuple_Type) && length(env.state) == 5
        obs, = convert(Tuple{T,Float64,Bool,Bool,PyDict}, env.state)
        obs
    elseif pyisinstance(env.state, PyCall.@pyglobalobj :PyTuple_Type) && length(env.state) == 4
        obs, = convert(Tuple{T,Float64,Bool,PyDict}, env.state)
        obs
    else
        convert(T, env.state)
    end
end

Random.seed!(env::GymEnv, s) = env.pyenv.seed(s)

# Base.display(env::GymEnv) = env.pyenv.render()

###
### utils
###
function space_transform(s::PyObject)
    spacetype = s.__class__.__name__
    if spacetype == "Box"
        ArrayProductDomain(map((l, h) -> l .. h, s.low, s.high))
    elseif spacetype == "Discrete"  # for GymEnv("CliffWalking-v0"), `s.n` is of type PyObject (numpy.int64)
        0:(py"int($s.n)"-1)
    elseif spacetype == "MultiBinary"
        ArrayProductDomain(fill(false:true, s.n))
    elseif spacetype == "MultiDiscrete"
        ArrayProductDomain(map(x -> 0:x-one(typeof(x)), s.nvec))
    elseif spacetype == "Tuple"
        TupleProductDomain(space_transform(x) for x in s.spaces)
    elseif spacetype == "Dict"
        NamedTupleProductDomain((k => space_transform(v) for (k, v) in s.spaces)...)
    else
        error("Don't know how to convert Gym Space of class [$(spacetype)]")
    end
end

function list_gym_env_names(;
    modules=[
        "gymnasium.envs.algorithmic",
        "gymnasium.envs.box2d",
        "gymnasium.envs.classic_control",
        "gymnasium.envs.mujoco",
        "gymnasium.envs.mujoco.ant_v3",
        "gymnasium.envs.mujoco.half_cheetah_v3",
        "gymnasium.envs.mujoco.hopper_v3",
        "gymnasium.envs.mujoco.humanoid_v3",
        "gymnasium.envs.mujoco.swimmer_v3",
        "gymnasium.envs.mujoco.walker2d_v3",
        "gymnasium.envs.robotics",
        "gymnasium.envs.toy_text",
        "gymnasium.envs.unittest",
        "d4rl.pointmaze",
        "d4rl.hand_manipulation_suite",
        "d4rl.gym_mujoco.gym_envs",
        "d4rl.locomotion.ant",
        "d4rl.gym_bullet.gym_envs",
        "d4rl.pointmaze_bullet.bullet_maze", # yet to include flow and carla
    ]
)
    if PyCall.pyexists("d4rl")
        pyimport("d4rl")
    end
    gym = pyimport("gymnasium")
    [x.id for x in values(gym.envs.registry) if split(x.entry_point, ':')[1] in modules]
end

"""
    install_gym(; packages = ["gymnasium", "pybullet"])
"""
function install_gym(; packages=["gymnasium", "pybullet"])
    # Use eventual proxy info
    proxy_arg = String[]
    if haskey(ENV, "http_proxy")
        push!(proxy_arg, "--proxy")
        push!(proxy_arg, ENV["http_proxy"])
    end
    # Import pip
    if !PyCall.pyexists("pip")
        # If it is not found, install it
        println("Pip not found on your system. Downloading it.")
        get_pip = joinpath(dirname(@__FILE__), "get-pip.py")
        download("https://bootstrap.pypa.io/get-pip.py", get_pip)
        run(`$(PyCall.python) $(proxy_arg) $get_pip --user`)
    end
    println("Installing required python packages using pip")
    run(`$(PyCall.python) $(proxy_arg) -m pip install --user --upgrade pip setuptools`)
    run(`$(PyCall.python) $(proxy_arg) -m pip install --user $(packages)`)
end
