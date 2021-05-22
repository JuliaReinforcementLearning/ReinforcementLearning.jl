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
1. `pkg> dev .`
1. `pkg> instantiate`
1. `pkg> up` (optional)

Then you can modify the code in your favourate editor and test it as usuall.
Sometimes, you may need to add some other dependencies. Remember to switch the
environment before adding new packages. For example, if you want to add
`Statistics` in `ReinforcementLearningBase`, first run `]activate
src/ReinforcementLearningBase`, then `]add Statistics`.

## How to contribute a new experiment?

Let's take a look at the folder structure of
`ReinforcementLearningExperiments.jl` first:

```
tree -d ./src/ReinforcementLearningExperiments

./src/ReinforcementLearningExperiments
├── deps
│   └── experiments
│       ├── assets
│       └── experiments
│           ├── CFR
│           ├── DQN
│           │   └── assets
│           ├── Offline
│           ├── Policy Gradient
│           │   └── assets
│           └── Search
├── src
│   └── experiments
└── test
```

We use the [DemoCards.jl](https://johnnychen94.github.io/DemoCards.jl/stable/)
to generate the documentation of all the experiments. If you want to contribute
a new experiment, simply create a `Your_Experiment.jl` file in a specific
algorithm category under the
`src/ReinforcementLearningExperiments/deps/experiments` folder.
Node that the this file should follow the format defined in
[Literate.jl](https://github.com/fredrikekre/Literate.jl).

!!! note
    And all the cells after the `#+ tangle=true` line will be extracted into the
    `ReinforcementLearningExperiments` package automatically.
