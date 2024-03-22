# Tips for Developers

## How to setup local development environment?

You can activate the local development mode as follows: from the base project directory, run:

```julia
using Pkg
Pkg.develop(path="src/ReinforcementLearningBase")
Pkg.develop(path="src/ReinforcementLearningCore")
Pkg.develop(path="src/ReinforcementLearningEnvironments")
Pkg.develop(path="src/ReinforcementLearningFarm") # optional
```

Sometimes, you may need to add some
extra dependencies. Remember to switch the environment before adding new
packages. For example, if you want to add
`Statistics` in `ReinforcementLearningBase`, first run `]activate
src/ReinforcementLearningBase`, then `]add Statistics`.

## How to enable debug timings for experiment runs?

Call `RLCore.TimerOutputs.enable_debug_timings(RLCore)` and default timings for hooks, policies and optimization steps will be printed. How do I reset the timer? Call `RLCore.TimerOutputs.reset_timer!(RLCore.timer)`. How do I show the timer results? Call `RLCore.timer`.

