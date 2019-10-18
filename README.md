![logo](.docs/src/assets/logo.png)

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaReinforcementLearning.github.io/ReinforcementLearning.jl/latest)
[![Build Status](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl.svg?branch=master)](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl)
[![codecov](https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearning.jl)

**ReinforcementLearning.jl**, as the name says, is a package for reinforcement learning research in Julia.

Our design principles are:

- **Reusability and extensibility**: Provide elaborately designed components and interfaces to help users implement new algorithms.
- **Easy experimentation**: Make it easy for new users to run benchmark experiments, compare different algorithms, evaluate and diagnose agents.
- **Reproducibility**: Facilitate reproducibility from traditional tabular methods to modern deep reinforcement learning algorithms.

Key capabilities/features include:

- Well tested traditional methods:
    - [`TDLearner`](@ref)
    - [`DifferentialTDLearner`](@ref)
    - [`TDλReturnLearner`](@ref)
    - [`DoubleLearner`](@ref)
    - [`MonteCarloLearner`](@ref)
    - [`GradientBanditLearner`](@ref)
    - [`ReinforcePolicy`](@ref)

- Efficiently implemented deep reinforcement learning algorithms:
    - Deep Q-Learning:
        - [`BasicDQNLearner`](@ref)
        - [`DQNLearner`](@ref)
        - [`PrioritizedDQNLearner`](@ref)
        - [`RainbowLearner`](@ref)

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