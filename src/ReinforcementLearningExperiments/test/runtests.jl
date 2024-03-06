using UUIDs
using Preferences

if Sys.isapple() && Sys.ARCH === :aarch64
    flux_uuid = UUID("587475ba-b771-5e3f-ad9e-33799f191a9c")
    set_preferences!(flux_uuid, "gpu_backend" => "Metal")

    using Metal
else
    using CUDA, cuDNN
    CUDA.allowscalar(false)
end

using Flux
using ReinforcementLearningExperiments
using Test

@info "Flux.GPU_BACKEND = $(Flux.GPU_BACKEND)"

experiments = [
    "JuliaRL_BasicDQN_CartPole",
    "JuliaRL_DQN_CartPole",
    "JuliaRL_DQNCartPole_GPU",
    "JuliaRL_IQN_CartPole",
    "JuliaRL_NFQ_CartPole",
    "JuliaRL_PrioritizedDQN_CartPole",
    "JuliaRL_QRDQN_CartPole",
    "JuliaRL_REMDQN_CartPole",
    "JuliaRL_Rainbow_CartPole"
]

for experiment_name in experiments
    @testset "$experiment_name" begin
        run(Experiment(experiment_name))
    end
end

deactivated_experiments = [
    "JuliaRL_VPG_CartPole",
    "JuliaRL_TRPO_CartPole",
    "JuliaRL_SAC_Pendulum",
    "JuliaRL_CQLSAC_Pendulum",
    "JuliaRL_MPODiscrete_CartPole",
    "JuliaRL_MPOContinuous_CartPole",
    "JuliaRL_MPOCovariance_CartPole",
    "JuliaRL_BC_CartPole",
    "JuliaRL_VMPO_CartPole",
    "JuliaRL_BasicDQN_MountainCar",
    "JuliaRL_DQN_MountainCar",
    "JuliaRL_A2C_CartPole",
    "JuliaRL_A2CGAE_CartPole",
    "JuliaRL_PPO_CartPole",
    "JuliaRL_MAC_CartPole",
    "JuliaRL_DDPG_Pendulum",
    "JuliaRL_TD3_Pendulum",
    "JuliaRL_PPO_Pendulum",
    "JuliaRL_BasicDQN_SingleRoomUndirected"
]
