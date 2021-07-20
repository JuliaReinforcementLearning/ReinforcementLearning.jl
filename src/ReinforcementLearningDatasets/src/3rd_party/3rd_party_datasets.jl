using PyCall
using StatsBase
using ArcadeLearningEnvironment:ROM_PATH
using Random:AbstractRNG
using ReinforcementLearningCore: SART, SARTS, BatchSampler

export get_dataset, env_names, install_packages

"""
    get_dataset(env_name, package)

This function fetches the dataset for the gym `env_name` from the `package` that is specified. This is intended to be a 3rd party wrapper that would return a NamedTuple.
The type of the NamedTuple depends on the dataset that is requested. For now d4rl package envs would return a NamedTuple{SARTS} and the rest would return a NamedTuple{SART}.

- `env_name` is the name of the gym env that the dataset is provided for.
- `package` is the name of the offline learning dataset package. For now supports `d4rl`, `d4rl_pybullet` and `d4rl_atari` for now.

`package` defaults to "d4rl" unless specified.

WARNING: It is advisable to use `atari` package envs with a sufficient amount of RAM because it rapidly fills up a huge amount of memory.
"""

# supports d4rl, pybullet and atari offline datasets
# TO-DO: get more info on mujoco envs
# TO-DO: give warning for loading atari datasets
function get_dataset(env_name::String, package::String = "d4rl")
    
    data = get_data(env_name, package)
        
    if package == "d4rl"
        return dataset = (state=(data["observations"]'),
                action=copy(data["actions"]'),
                reward=copy(data["rewards"]'),
                terminal=copy(data["terminals"]'),
                next_state=copy(data["next_observations"]'),)
    elseif package in ["d4rl_pybullet", "d4rl_atari"]
        dataset = (state=copy(data["observations"]'),
                    action=copy(data["actions"]'),
                    reward=copy(data["rewards"]'),
                    terminal=copy(data["terminals"]'),)
        return dataset
    else 
        error(
        "The package specified is not supported or does not exist"
        )
    end

end

# Intended as a helper function that would fetch the data for the env and package that is specified before packaging it into a NamedTuple.
function get_data(env_name::String, package::String) 
    
    if !PyCall.pyexists("gym")
        error(
            "Cannot import module 'gym'.\n\nIf you did not yet install it, try running\n`ReinforcementLearningEnvironments.install_gym()`\n",
        )
    end
    
    gym =  pyimport("gym")
    
    if !PyCall.pyexists(package)
            error(
                "Cannot import module $(package), try using install_packages",
            )
    end
    
    if package == "d4rl"
        d4rl = pyimport(package)
        env = gym.make(env_name)
        return d4rl.qlearning_dataset(env)
    end
    
    pyimport(package)
    
    if ! (env_name in env_names(package))
        error(
            "Dataset for this environment doesn't exist in package"
        )
    end
    
    env = gym.make(env_name)
        
    return env.get_dataset()
    
end

"""
    env_names(package)

Fetches the list of `env`s that are available for the specified `package`.
Supported packages are:

- `d4rl`
- `d4rl_pybullet`
- `d4rl_atari`
"""
function env_names(package::String)
    
    if !PyCall.pyexists("gym")
        error(
            "Cannot import module 'gym'.\n\nIf you did not yet install it, try running\n`ReinforcementLearningEnvironments.install_gym()`\n",
        )
    end
    
    gym =  pyimport("gym")
    
    pyimport(package)
    
    modules = [
            "d4rl.gym_mujoco.gym_envs"
            "d4rl_pybullet.envs"
            "d4rl_atari.envs"
            ]
    
    if package == "d4rl" mod = modules[1] end
    if package == "d4rl_pybullet" mod = modules[2] end
    if package == "d4rl_atari" mod = modules[3] end
    
    # maybe we can check once instead of calling it everytime when we use atari_dataset
    if package == "d4rl_atari"
        run(`$(PyCall.python) -m atari_py.import_roms $(ROM_PATH)`);
    end
    
    [x.id for x in gym.envs.registry.all() if split(x.entry_point, ':')[1] == mod]
end

# Check if there are other dependencies on the package
# Make sure that you install mujoco and mujoco-py before installing the below packages
"""
    install_packages(packages)

Installs the specified packages using pip. Still under development and not tested on other configurations.
"""
function install_packages(packages = ["d4rl_pybullet", "d4rl_atari", "d4rl"])
    proxy_arg = String[]
    if haskey(ENV, "http_proxy")
        push!(proxy_arg, "--proxy")
        push!(proxy_arg, ENV["http_proxy"])
    end
    
    if "d4rl" in packages
        #installation of d4rl
        print("installing d4rl using pip")
        if !("d4rl" in readdir()) run(`git clone https://github.com/rail-berkeley/d4rl.git`) end
        cd("d4rl")
        run(`$(PyCall.python) $(proxy_arg) -m pip install -e .`) 
    end
    
    if "d4rl_pybullet" in packages
        #installation of d4rl_pybullet
        print("installing d4rl_pybullet")
        run(`$(PyCall.python) $(proxy_arg) -m pip install git+https://github.com/takuseno/d4rl-pybullet`)
    end
    
    if "d4rl_atari" in packages
    #installation of d4rl_atari
        print("installing d4rl_atari")
        run(`$(PyCall.python) $(proxy_arg) -m pip install atari-py`)
        run(`$(PyCall.python) $(proxy_arg) -m pip install git+https://github.com/takuseno/d4rl-atari`)
    end
end

function StatsBase.sample(rng::T, t::NamedTuple{SARTS}, s::BatchSampler{SARTS}) where T<:AbstractRNG
    inds = rand(rng, 1:length(t), s.batch_size)
    NamedTuple{SARTS}(@view dataset[x][:, inds] for x in SARTS)
end

function StatsBase.sample(rng::T, t::NamedTuple{SART}, s::BatchSampler{SART}) where T<:AbstractRNG
    inds = rand(rng, 1:length(t), s.batch_size)
    NamedTuple{SART}(@view dataset[x][:, inds] for x in SART)
end