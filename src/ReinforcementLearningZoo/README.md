<div align="center"> 
<a href="https://en.wikipedia.org/wiki/Tangram"> <img src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Tangram-man.svg" width="200"> </a>
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

- DQN
- Prioritized DQN
- Rainbow
- IQN
- A2C
- PPO
- DDPG

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
- ``E`JuliaRL_A2CGAE_CartPole` ``
- ``E`JuliaRL_PPO_CartPole` ``
- ``E`JuliaRL_DDPG_Pendulum` ``
- ``E`JuliaRL_BasicDQN_MountainCar` `` (Thanks to [@felixchalumeau](https://github.com/felixchalumeau))
- ``E`JuliaRL_DQN_MountainCar` `` (Thanks to [@felixchalumeau](https://github.com/felixchalumeau))
- ``E`Dopamine_DQN_Atari(pong)` ``
- ``E`Dopamine_Rainbow_Atari(pong)` ``
- ``E`Dopamine_IQN_Atari(pong)` ``
- ``E`rlpyt_A2C_Atari(pong)` ``
- ``E`rlpyt_PPO_Atari(pong)` ``

**Notes:**

- Experiments on `CartPole` usually run faster with CPU only due to the overhead of sending data between CPU and GPU.
- It shouldn't surprise you that our experiments on `CartPole` are much faster than those written in Python. The secret is that our environment is written in Julia!
- Remember to set `JULIA_NUM_THREADS` to enable multi-threading when using algorithms like `A2C` and `PPO`.
- Experiments on `Atari` are only available when you have `ArcadeLearningEnvironment.jl` installed and `using ArcadeLearningEnvironment`.
- Different configurations might affect the performance a lot. According to our tests, our implementations are generally comparable to those written in PyTorch or TensorFlow with the same configuration (sometimes we are significantly faster).