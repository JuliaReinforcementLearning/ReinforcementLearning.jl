# Tips for Developers

## How to setup local development environment?

You can activate the local development mode as follows: from the base project directory,
load `ReinforcementLearning` via `using ReinforcementLearning`.
Then run `ReinforcementLearning.activate_devmode!()`.
Sometimes, you may need to add some
extra dependencies. Remember to switch the environment before adding new
packages. For example, if you want to add
`Statistics` in `ReinforcementLearningBase`, first run `]activate
src/ReinforcementLearningBase`, then `]add Statistics`.

## How to contribute a new experiment?

We use the [DemoCards.jl](https://johnnychen94.github.io/DemoCards.jl/stable/)
to generate the documentation of all the experiments. If you want to contribute
a new experiment, simply create a `Your_Experiment.jl` file in a specific
algorithm category under the `docs/experiments` folder.
Note that this file should follow the format defined in
[Literate.jl](https://github.com/fredrikekre/Literate.jl). And then update the
`config.json` file correspondingly. If your experiment needs an extra
dependency, remember to update both `docs/Project.toml` and
`src/ReinforcementLearningExperiments/Project.toml`.

!!! note
    All the cells after the `#+ tangle=true` line in `Your_Experment.jl` will be extracted into the
    `ReinforcementLearningExperiments` package automatically. This feature is
    supported by [Weave.jl](https://weavejl.mpastell.com/stable/usage/#tangle).

## How to enable debug timings for experiment runs?

Call `RLCore.TimerOutputs.enable_debug_timings(RLCore)` and default timings for hooks, policies and optimization steps will be printed. How do I reset the timer? Call `RLCore.TimerOutputs.reset_timer!(RLCore.timer)`. How do I show the timer results? Call `RLCore.timer`.

