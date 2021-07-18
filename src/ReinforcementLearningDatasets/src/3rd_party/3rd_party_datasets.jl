using PyCall
using StatsBase
using ArcadeLearningEnvironment:ROM_PATH
using Random:AbstractRNG
using ReinforcementLearning: SART, SARTS, BatchSampler

export get_dataset, env_names, install_packages

# supports d4rl, pybullet and atari offline datasets
# TO-DO: get more info on mujoco envs
# TO-DO: give warning for getting atari datasets
function get_dataset(env_name::String, package::String)
    
    if !PyCall.pyexists("gym")
        error(
            "Cannot import module 'gym'.\n\nIf you did not yet install it, try running\n`ReinforcementLearningEnvironments.install_gym()`\n",
        )
    end
    
    gym =  pyimport("gym")
    
    data = get_data(env_name, package)
        
    if package == "d4rl"
        dataset = (state=(data["observations"]'),
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
        return data = d4rl.qlearning_dataset(env)
    end
    
    # Needs to import ROM path
    if package == "d4rl_atari"
        run(`$(PyCall.python) -m atari_py.import_roms $(ROM_PATH)`);
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

function env_names(package::String)
    if !PyCall.pyexists("gym")
        error(
            "Cannot import module 'gym'.\n\nIf you did not yet install it, try running\n`ReinforcementLearningEnvironments.install_gym()`\n",
        )
    end
    
    gym =  pyimport("gym")
    
    pyimport(package)
    modules = [
        "d4rl_pybullet.envs"
        "d4rl.gym_mujoco.gym_envs"
        "d4rl_atari.envs"
    ]
    if package == "d4rl_atari"
        run(`$(PyCall.python) -m atari_py.import_roms $(ROM_PATH)`);
    end
    
    [x.id for x in gym.envs.registry.all() if split(x.entry_point, ':')[1] in modules]
end

# Check if there are other dependencies on the package
# Make sure that you install mujoco and mujocopy before installing the below packages
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