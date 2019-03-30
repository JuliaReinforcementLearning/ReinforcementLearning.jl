# Benchmarks of the runtime for different environments

Each environment is estimated to run **1000** steps.

| Environment | mean time | median time | memory | allocs |
| :---------- | --------: | ----------: | -----: | -----: |
|HanabiEnv()|37.900 ms|36.825 ms|2.98 MiB|7999|
|basic_ViZDoom_env()|720.176 ms|723.255 ms|216.30 KiB|4011|
|CartPoleEnv()|147.698 μs|126.692 μs|41.75 KiB|1096|
|MountainCarEnv()|114.707 μs|98.710 μs|31.25 KiB|1000|
|PendulumEnv()|147.294 μs|115.320 μs|142.31 KiB|2018|
|MDPEnv(LegacyGridWorld())|258.661 μs|195.653 μs|250.00 KiB|5000|
|POMDPEnv(TigerPOMDP())|1.164 ms|1.041 ms|250.00 KiB|9000|
|SimpleMDPEnv()|1.178 ms|1.055 ms|281.25 KiB|10000|
|deterministic_MDP()|14.340 ms|13.846 ms|310.25 KiB|11856|
|absorbing_deterministic_tree_MDP()|2.616 ms|2.472 ms|294.69 KiB|10860|
|stochastic_MDP()|1.339 ms|1.252 ms|281.25 KiB|10000|
|stochastic_tree_MDP()|1.443 ms|1.278 ms|281.25 KiB|10000|
|deterministic_tree_MDP_with_rand_reward()|1.872 ms|1.633 ms|281.25 KiB|10000|
|deterministic_tree_MDP()|1.885 ms|1.622 ms|281.25 KiB|10000|
|deterministic_MDP()|14.120 ms|13.774 ms|310.28 KiB|11858|
|AtariEnv("pong")|1.172 s|1.178 s|46.89 KiB|2001|
|GymEnv("CliffWalking-v0")|26.581 ms|24.427 ms|640.78 KiB|23003|
|GymEnv("KellyCoinflip-v0")|22.171 ms|19.117 ms|627.38 KiB|22933|
|GymEnv("FrozenLake8x8-v0")|28.953 ms|27.275 ms|643.91 KiB|23063|
|GymEnv("FrozenLake-v0")|30.920 ms|29.430 ms|658.28 KiB|23339|
|GymEnv("Pendulum-v0")|62.122 ms|57.287 ms|1.62 MiB|46018|
|GymEnv("CubeCrashScreenBecomesBlack-v0")|58.252 ms|55.065 ms|1.47 MiB|39099|
|GymEnv("RepeatCopy-v0")|40.768 ms|38.163 ms|1.00 MiB|32083|
|GymEnv("DuplicatedInput-v0")|36.404 ms|33.300 ms|1.01 MiB|32311|
|GymEnv("MemorizeDigits-v0")|58.699 ms|54.578 ms|1.47 MiB|39117|
|GymEnv("CubeCrashSparse-v0")|49.924 ms|45.917 ms|1.47 MiB|39099|
|GymEnv("ReversedAddition3-v0")|40.451 ms|37.556 ms|1017.34 KiB|31933|
|GymEnv("HotterColder-v0")|43.337 ms|39.682 ms|1.13 MiB|34018|
|GymEnv("Reverse-v0")|39.199 ms|36.470 ms|1018.75 KiB|31960|
|GymEnv("GuessingGame-v0")|35.782 ms|32.120 ms|1.13 MiB|34018|
|GymEnv("MountainCarContinuous-v0")|34.140 ms|28.901 ms|1.62 MiB|46006|
|GymEnv("Blackjack-v0")|54.012 ms|50.460 ms|671.72 KiB|23097|
|GymEnv("CartPole-v1")|32.616 ms|29.292 ms|1.12 MiB|35108|
|GymEnv("CubeCrash-v0")|59.660 ms|52.390 ms|1.47 MiB|39099|
|GymEnv("ReversedAddition-v0")|38.326 ms|35.061 ms|1017.81 KiB|31942|
|GymEnv("MountainCar-v0")|36.495 ms|32.956 ms|1.11 MiB|35018|
|GymEnv("NChain-v0")|14.355 ms|12.568 ms|640.94 KiB|23006|
|GymEnv("Roulette-v0")|18.185 ms|15.033 ms|643.44 KiB|23054|
|GymEnv("Acrobot-v1")|212.436 ms|210.213 ms|1.11 MiB|35009|
|GymEnv("Copy-v0")|34.270 ms|31.824 ms|1.00 MiB|32101|
|GymEnv("Taxi-v2")|29.261 ms|27.130 ms|641.56 KiB|23018|
|GymEnv("CartPole-v0")|32.077 ms|28.078 ms|1.12 MiB|35108|
