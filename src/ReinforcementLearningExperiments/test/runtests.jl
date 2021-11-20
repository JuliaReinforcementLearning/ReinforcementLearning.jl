using ReinforcementLearningExperiments
using CUDA

CUDA.allowscalar(false)

run(E`JuliaRL_BasicDQN_CartPole`)
run(E`JuliaRL_BC_CartPole`)
run(E`JuliaRL_DQN_CartPole`)
run(E`JuliaRL_PrioritizedDQN_CartPole`)
run(E`JuliaRL_Rainbow_CartPole`)
run(E`JuliaRL_QRDQN_CartPole`)
run(E`JuliaRL_REMDQN_CartPole`)
run(E`JuliaRL_IQN_CartPole`)
run(E`JuliaRL_VMPO_CartPole`)
run(E`JuliaRL_VPG_CartPole`)
run(E`JuliaRL_BasicDQN_MountainCar`)
run(E`JuliaRL_DQN_MountainCar`)
run(E`JuliaRL_A2C_CartPole`)
run(E`JuliaRL_A2CGAE_CartPole`)
run(E`JuliaRL_PPO_CartPole`)
run(E`JuliaRL_MAC_CartPole`)
run(E`JuliaRL_DDPG_Pendulum`)
run(E`JuliaRL_SAC_Pendulum`)
run(E`JuliaRL_TD3_Pendulum`)
run(E`JuliaRL_PPO_Pendulum`)

run(E`JuliaRL_BasicDQN_SingleRoomUndirected`)
