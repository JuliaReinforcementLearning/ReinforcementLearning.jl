export d4rl_dataset_params

function d4rl_pybullet_dataset_params()
    dataset = keys(D4RL_PYBULLET_URLS)
    repo = "d4rl-pybullet"
    @info dataset repo
end

const D4RL_PYBULLET_URLS = Dict(
    "hopper-bullet-mixed-v0" => "https://www.dropbox.com/s/xv3p0h7dzgxt8xb/hopper-bullet-mixed-v0.hdf5?dl=1", 
    "walker2d-bullet-random-v0" => "https://www.dropbox.com/s/1gwcfl2nmx6878m/walker2d-bullet-random-v0.hdf5?dl=1", 
    "hopper-bullet-medium-v0" => "https://www.dropbox.com/s/w22kgzldn6eng7j/hopper-bullet-medium-v0.hdf5?dl=1", 
    "walker2d-bullet-mixed-v0" => "https://www.dropbox.com/s/i4u2ii0d85iblou/walker2d-bullet-mixed-v0.hdf5?dl=1",
    "halfcheetah-bullet-mixed-v0" => "https://www.dropbox.com/s/scj1rqun963aw90/halfcheetah-bullet-mixed-v0.hdf5?dl=1", 
    "halfcheetah-bullet-random-v0" => "https://www.dropbox.com/s/jnvpb1hp60zt2ak/halfcheetah-bullet-random-v0.hdf5?dl=1",
    "walker2d-bullet-medium-v0" => "https://www.dropbox.com/s/v0f2kz48b1hw6or/walker2d-bullet-medium-v0.hdf5?dl=1", 
    "hopper-bullet-random-v0" => "https://www.dropbox.com/s/bino8ojd7iq4p4d/hopper-bullet-random-v0.hdf5?dl=1", 
    "ant-bullet-random-v0" => "https://www.dropbox.com/s/2xpmh4wk2m7i8xh/ant-bullet-random-v0.hdf5?dl=1", 
    "halfcheetah-bullet-medium-v0" => "https://www.dropbox.com/s/v4xgssp1w968a9l/halfcheetah-bullet-medium-v0.hdf5?dl=1", 
    "ant-bullet-medium-v0" => "https://www.dropbox.com/s/6n79kwd94xthr1t/ant-bullet-medium-v0.hdf5?dl=1", 
    "ant-bullet-mixed-v0" => "https://www.dropbox.com/s/pmy3dzab35g4whk/ant-bullet-mixed-v0.hdf5?dl=1"
)

function d4rl_pybullet_init()
    repo = "d4rl-pybullet"
    for ds in keys(D4RL_PYBULLET_URLS)
        register(
            DataDep(
                repo* "-" * ds,
                """
                Credits: https://github.com/takuseno/d4rl-pybullet
                The following dataset is fetched from the d4rl-pybullet. 
                """, 
                D4RL_PYBULLET_URLS[ds],
            )
        )
    end
    nothing
end