using ReinforcementLearningExperiments
using CUDA

using Requires



using Requires



CUDA.allowscalar(false)

run(E`JuliaRL_NFQ_CartPole`)
run(E`JuliaRL_BasicDQN_CartPole`)
run(E`JuliaRL_DQN_CartPole`)
run(E`JuliaRL_PrioritizedDQN_CartPole`)
run(E`JuliaRL_QRDQN_CartPole`)
run(E`JuliaRL_REMDQN_CartPole`)
run(E`JuliaRL_IQN_CartPole`)
run(E`JuliaRL_Rainbow_CartPole`)
run(E`JuliaRL_VPG_CartPole`)
run(E`JuliaRL_MPODiscrete_CartPole`)
run(E`JuliaRL_MPOContinuous_CartPole`)
run(E`JuliaRL_MPOCovariance_CartPole`)
run(E`JuliaRL_IDQN_TicTacToe`)
@require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" run(E`JuliaRL_DQN_MPESimple`)
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
