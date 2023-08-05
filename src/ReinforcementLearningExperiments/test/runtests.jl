using ReinforcementLearningExperiments
using UUIDs
using Preferences

if Sys.isapple()
    flux_uuid = UUID("587475ba-b771-5e3f-ad9e-33799f191a9c")
    set_preferences!(flux_uuid, "gpu_backend" => "Metal")

    using Metal
else
    using CUDA, cuDNN
    CUDA.allowscalar(false)
end

using Flux
println("Flux.GPU_BACKEND = $(Flux.GPU_BACKEND)")

run(E`JuliaRL_NFQ_CartPole`)
run(E`JuliaRL_BasicDQN_CartPole`)
run(E`JuliaRL_DQN_CartPole`)
# run(E`JuliaRL_PrioritizedDQN_CartPole`)
run(E`JuliaRL_QRDQN_CartPole`)
run(E`JuliaRL_REMDQN_CartPole`)
run(E`JuliaRL_IQN_CartPole`)
run(E`JuliaRL_Rainbow_CartPole`)
# run(E`JuliaRL_VPG_CartPole`)
run(E`JuliaRL_MPODiscrete_CartPole`)
run(E`JuliaRL_MPOContinuous_CartPole`)
run(E`JuliaRL_MPOCovariance_CartPole`)
run(E`JuliaRL_DQNCartPole_GPU`)
# run(E`JuliaRL_BC_CartPole`)
# run(E`JuliaRL_VMPO_CartPole`)
# run(E`JuliaRL_BasicDQN_MountainCar`)
# run(E`JuliaRL_DQN_MountainCar`)
# run(E`JuliaRL_A2C_CartPole`)
# run(E`JuliaRL_A2CGAE_CartPole`)
# run(E`JuliaRL_PPO_CartPole`)
# run(E`JuliaRL_MAC_CartPole`)
# run(E`JuliaRL_DDPG_Pendulum`)
# run(E`JuliaRL_SAC_Pendulum`)
# run(E`JuliaRL_TD3_Pendulum`)
# run(E`JuliaRL_PPO_Pendulum`)

# run(E`JuliaRL_BasicDQN_SingleRoomUndirected`)
