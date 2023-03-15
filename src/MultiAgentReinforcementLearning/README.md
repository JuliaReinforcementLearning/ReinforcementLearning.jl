# MultiAgentReinforcementLearning.jl

This package provides multi agent reinforcement learning baselines as well as some further techniques.

Basically, the following parts are included in this module

- `MultiAgentManager`
- `MADDPG`
- `DIAL`
- `CoordinationGraphs`
- `MessageContainer` for `ReinforcementLearningTrajectory` Module
- `WholeEpisodeSampler` for `ReinforcementLearningTrajectory` Module

The baseline methods - namely `MultiAgentManager` and `MADDPG` were already present in the `ReinforcementLearningZoo` module and were removed previously. In order to provide baselines for Multi Agent Reinforcement Learning, they were adjusted to the changes of the library since removal. The adapted source code were thankfully provided by Peter Chen from the [ECNU](https://english.ecnu.edu.cn/). His [blog post](https://juliareinforcementlearning.org/blog/ospp_report_210370190/) describes his work on the module and some outstanding adjustments to the library:

- `Neural Fictitious Self-play(NFSP)` algorithm
- `Exploitability Descent(ED)` algorithm
