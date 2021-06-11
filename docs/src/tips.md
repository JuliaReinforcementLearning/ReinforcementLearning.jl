# Tips for Developers

## How to setup local development environment?

`ReinforcementLearning.jl` is kind of different from most packages you've seen.
It simply re-export all the names in its dependent packages. So when you execute
`]dev ReinforcementLearning` in the Julia REPL, all the dependents are not
turned into the development mode automatically. The recommended process is:

1. `git clone git@github.com:JuliaReinforcementLearning/ReinforcementLearning.jl.git`
1. `cd ReinforcementLearning.jl`
1. `julia --project`
1. Press `]` to enter package mode in the Julia REPL.
1. `pkg> instantiate`
1. `pkg> up` (optional)

Then you can modify the code in your favorite editor and test it as usual.
Sometimes, you may need to add some extra dependencies. Remember to switch the
environment before adding new packages. For example, if you want to add
`Statistics` in `ReinforcementLearningBase`, first run `]activate
src/ReinforcementLearningBase`, then `]add Statistics`.

If you'd like to test your modifications in some other environments, remember to
run `dev /path/to/ReinforcementLearningBase` (or any other subpackages you've modified).

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
