<div align="center">
<a href="https://en.wikipedia.org/wiki/Tangram"> <img src="./docs/logo/logo.gif"> </a>
<p> <a href="https://wiki.c2.com/?MakeItWorkMakeItRightMakeItFast">"Make It Work Make It Right Make It Fast"</a></p>
<p>― <a href="https://wiki.c2.com/?KentBeck">KentBeck</a></p>
</div>

<hr>
<p align="center">
  <a href="https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl">
  <img src="https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl.svg?branch=master">
  </a>
</p>


This project aims to provide some implementations of the most typical reinforcement learning algorithms.

# Algorithms Implemented
- VPG (Vanilla Policy Gradient, with a baseline)
- DQN
- Prioritized DQN
- Rainbow
- IQN
- A2C
- PPO
- DDPG
- TD3
- SAC
- CFR/OS-MCCFR/ES-MCCFR
- Minimax

If you are looking for tabular reinforcement learning algorithms, you may refer [ReinforcementLearningAnIntroduction.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl).

# Built-in Experiments

Some built-in experiments are exported to help new users to easily run benchmarks with one line (for example, ``run(E`JuliaRL_BasicDQN_CartPole`)``). For experienced users, you are suggested to check [the source code](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/tree/master/src/experiments) of those experiments and make changes as needed.

## List of built-in experiments

- ``E`JuliaRL_BasicDQN_CartPole` ``
- ``E`JuliaRL_DQN_CartPole` ``
- ``E`JuliaRL_PrioritizedDQN_CartPole` ``
- ``E`JuliaRL_Rainbow_CartPole` ``
- ``E`JuliaRL_IQN_CartPole` ``
- ``E`JuliaRL_A2C_CartPole` ``
- ``E`JuliaRL_A2CGAE_CartPole` `` (Thanks to [@sriram13m](https://github.com/sriram13m))
- ``E`JuliaRL_PPO_CartPole` ``
- ``E`JuliaRL_VPG_CartPole` `` (Thanks to [@norci](https://github.com/norci))
- ``E`JuliaRL_VPG_Pendulum` `` (continuous action space)
- ``E`JuliaRL_VPG_PendulumD` `` (discrete action space)
- ``E`JuliaRL_DDPG_Pendulum` ``
- ``E`JuliaRL_TD3_Pendulum` `` (Thanks to [@rbange](https://github.com/rbange))
- ``E`JuliaRL_SAC_Pendulum` `` (Thanks to [@rbange](https://github.com/rbange))
- ``E`JuliaRL_PPO_Pendulum` ``
- ``E`JuliaRL_BasicDQN_MountainCar` `` (Thanks to [@felixchalumeau](https://github.com/felixchalumeau))
- ``E`JuliaRL_DQN_MountainCar` `` (Thanks to [@felixchalumeau](https://github.com/felixchalumeau))
- ``E`JuliaRL_Minimax_OpenSpiel(tic_tac_toe)` ``
- ``E`JuliaRL_TabularCFR_OpenSpiel(kuhn_poker)` ``
- ``E`JuliaRL_DQN_SnakeGame` ``
- ``E`Dopamine_DQN_Atari(pong)` ``
- ``E`Dopamine_Rainbow_Atari(pong)` ``
- ``E`Dopamine_IQN_Atari(pong)` ``
- ``E`rlpyt_A2C_Atari(pong)` ``
- ``E`rlpyt_PPO_Atari(pong)` ``

### Notes:

- Experiments on `CartPole` usually run faster with CPU only due to the overhead of sending data between CPU and GPU.
- It shouldn't surprise you that our experiments on `CartPole` are much faster than those written in Python. The secret is that our environment is written in Julia!
- Remember to set `JULIA_NUM_THREADS` to enable multi-threading when using algorithms like `A2C` and `PPO`.
- Experiments on `Atari` (`OpenSpiel`, `SnakeGame`) are only available after you have `ArcadeLearningEnvironment.jl` (`OpenSpiel.jl`, `SnakeGame.jl`) installed and `using ArcadeLearningEnvironment` (`using OpenSpiel`, `using SnakeGame`).

### Speed

- Different configurations might affect the performance a lot. According to our tests, our implementations are generally comparable to those written in PyTorch or TensorFlow with the same configuration (sometimes we are significantly faster).

The following data are collected from experiments on *Intel(R) Xeon(R) W-2123 CPU @ 3.60GHz* with a GPU card of *RTX 2080ti*.

| Experiment | FPS | Notes |
|:---------- |:----:| ----:|
| ``E`Dopamine_DQN_Atari(pong)` `` | ~210 | Use the same config of [dqn.gin in google/dopamine](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin)|
| ``E`Dopamine_Rainbow_Atari(pong)` `` | ~171 | Use the same config of [rainbow.gin in google/dopamine](https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/configs/rainbow.gin)|
| ``E`Dopamine_IQN_Atari(pong)` `` | ~162 | Use the same config of [implicit_quantile.gin in google/dopamine](https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/configs/implicit_quantile.gin)|
|``E`rlpyt_A2C_Atari(pong)` ``| ~768 | Use the same default parameters of [A2C in rlpyt](https://github.com/astooke/rlpyt/blob/master/rlpyt/algos/pg/a2c.py) with **4 threads**|
| ``E`rlpyt_PPO_Atari(pong)` `` | ~711 | Use the same default parameters of [PPO in rlpyt](https://github.com/astooke/rlpyt/blob/master/rlpyt/algos/pg/ppo.py) with **4 threads**|
