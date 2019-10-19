<div align="center">
  <p>
  <img src="https://raw.githubusercontent.com/JuliaReinforcementLearning/ReinforcementLearning.jl/master/docs/src/assets/logo.png">
  </p>

  <p>
  <a href="https://JuliaReinforcementLearning.github.io/ReinforcementLearning.jl/latest"><img src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
  <a href="https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl"><img src="https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl.svg?branch=master"></a>
  <a href="https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearning.jl"><img src="https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearning.jl/branch/master/graph/badge.svg"></a>
  </p>
</div>

**ReinforcementLearning.jl**, as the name says, is a package for reinforcement learning research in Julia.

Our design principles are:

- **Reusability and extensibility**: Provide elaborately designed components and interfaces to help users implement new algorithms.
- **Easy experimentation**: Make it easy for new users to run benchmark experiments, compare different algorithms, evaluate and diagnose agents.
- **Reproducibility**: Facilitate reproducibility from traditional tabular methods to modern deep reinforcement learning algorithms.

Key capabilities/features include:

- Well tested traditional methods:
    - [`TDLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.TDLearner)
    - [`DifferentialTDLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.DifferentialTDLearner)
    - [`TDλReturnLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.TDλReturnLearner)
    - [`DoubleLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.DoubleLearner)
    - [`MonteCarloLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.MonteCarloLearner)
    - [`GradientBanditLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.GradientBanditLearner)
    - [`ReinforcePolicy`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/policies/#ReinforcementLearning.ReinforcePolicy)

- Efficiently implemented deep reinforcement learning algorithms:
    - Deep Q-Learning:
        - [`BasicDQNLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.BasicDQNLearner)
        - [`DQNLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.DQNLearner)
        - [`PrioritizedDQNLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.PrioritizedDQNLearner)
        - [`RainbowLearner`](https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/components/learners/#ReinforcementLearning.RainbowLearner)

- Pluggable deep learning framework backend:
    - [Flux.jl](https://github.com/FluxML/Flux.jl)
    - [Knet.jl](https://github.com/denizyuret/Knet.jl)

- Built-in [TensorBoard](https://github.com/PhilipVinc/TensorBoardLogger.jl) support.


## Installation

This package can be installed from the package manager in Julia's REPL:

```
] add ReinforcementLearning
```

> **Warning**: Considering that this package is still under rapid development, you're strongly suggested to read the [Documentation](https://JuliaReinforcementLearning.github.io/ReinforcementLearning.jl/latest) for how to use the latest code.