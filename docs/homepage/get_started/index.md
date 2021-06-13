@def title = "A Whirlwind Tour of ReinforcementLearning.jl"
@def description = "Welcome to the world of reinforcement learning in Julia. Now let's get started in 3 lines!"
@def is_enable_toc = false
@def has_code = true
@def has_math = true

@def front_matter = """
    {
        "authors": [
            {
                "author":"Jun Tian",
                "authorURL":"https://github.com/findmyway",
                "affiliation":"",
                "affiliationURL":""
            }
        ],
        "publishedDate":"2021-01-26",
        "citationText":"Jun Tian, 2021"
    }"""

@def appendix = """
    ### Corrections
    If you see mistakes or want to suggest changes, please [create an issue](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues) in the source repository.
    """

@def bibliography = "bibliography.bib"

@def RL_VERSION = string([v.version for (k,v) in Pkg.dependencies() if v.name=="ReinforcementLearning"][1])
@def JULIA_VERSION = string(VERSION)

## Prepare

First things first, [download](https://julialang.org/downloads/) and install
Julia of the latest stable version. ReinforcementLearning.jl is tested on all
platforms, so just choose the one you are familiar with. If you already have
Julia installed, please make sure that it is {{ fill JULIA_VERSION }} or above.

\aside{ReinforcementLearning.jl relies on some features introduced since `v1.3`,
like
[MultiThreading](https://docs.julialang.org/en/v1/base/multi-threading/index.html),
and [Artifacts](https://julialang.github.io/Pkg.jl/dev/artifacts/)}

Another useful tool is [tensorboard](https://github.com/tensorflow/tensorboard)
\footnote{You don't need to install the whole TensorFlow to use the TensorBoard.
Behind the scene, ReinforcementLearning.jl uses
[TensorBoardLogger.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl) to
write data into the format that TensorBoard recognizes.}. You can install it via
`pip install tensorboard` with the python package installer
[`pip`](https://pip.pypa.io/en/stable/installing/).

## Get Started

Run `julia` in the command line (or double-click the Julia executable) and now
you are in an interactive session (also known as a read-eval-print loop or
"REPL"). Then execute the following code: 

```julia
] add ReinforcementLearning

using ReinforcementLearning

run(E`JuliaRL_BasicDQN_CartPole`)
```

So what's happening here?

1. In the first line, typing `]` will bring you to the *Pkg* mode. `add
   ReinforcementLearning` will install the latest version of
   `ReinforcementLearning.jl` for you. And then remember to press backspace or
   ^C to get back to the normal mode. All examples in this website are built with
   `ReinforcementLearning` of version {{ fill RL_VERSION}} . Note that sometimes
   you may have an old version installed. The reason is that some of the
   packages you have installed in your current Julia environment have an
   outdated dependency, resulting in a downgraded install of
   `ReinforcementLearning.jl`. You can confirm it by installing the latest
   master branch with `] add ReinforcementLearning#master`. To solve this
   problem, you can create a temporary directory and then activate the Julia
   environment there with `] activate /path/to/tmp/dir`.

1. `using ReinforcementLearning` will bring the names exported in
   `ReinforcementLearning` into global scope. If this is your first time to run,
   you'll see *precompiling ReinforcementLearning*. And it may take a while.

1. The third line means, `run` a predefined **E**xperiment named `JuliaRL_BasicDQN_CartPole` \footnote{The ``E`JuliaRL_BasicDQN_CartPole` `` is a handy [command literal](https://docs.julialang.org/en/v1/manual/metaprogramming/index.html#Non-Standard-String-Literals-1) to instantiate a prebuilt experiment.}.

CartPole is considered to be one of the simplest environments for DRL (Deep
Reinforcement Learning) algorithms testing. The state of the CartPole
environment can be described with 4 numbers and the actions are two integers(`1`
and `2`). Before game terminates, agent can gain a reward of `+1` for each step.
By default, the game will be forced to terminate after 200 steps, thus the
maximum reward of an episode is `200`. 

While the experiment is running, you'll see the following information and a
progress bar. The information may be slightly different based on your platform
and your current working directory. Note that the first run would be slow. On a
modern computer, the experiment should be finished in a minute.

```julia:./display_JuliaRL_BasicDQN_CartPole_1
#hideall
using ReinforcementLearning
e = E`JuliaRL_BasicDQN_CartPole`
print(e.description)
```

\output{./display_JuliaRL_BasicDQN_CartPole_1}

```julia:./display_JuliaRL_BasicDQN_CartPole_2
#hideall
println(e)
```

Follow the instruction above and run `tensorboard --logdir /the/path/shown/above`, then a link will be prompted (typically it's `http://YourHost:6006/`). Now open it in your browser, you'll see a webpage similar to the following one:

\dfig{page;tensorboard_demo.png;Here two important variables are logged: training **loss** per update and total **reward** of each episode during training. As you can see, our agent can reach the maximum reward after training for about 4k steps.}

## Exercise

Now that you already know how to run the experiment of
[BasicDQN](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_zoo/#ReinforcementLearningZoo.BasicDQNLearner)
algorithm with the CartPole environment. You are suggested to try some other
experiments below to compare the performance of different algorithms
\footnote{For the full list of supported algorithms, please visit
[ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl#list-of-built-in-experiments)}:


\aside{Note that the parameters in the experiments listed here are tuned.}

- ``E`JuliaRL_BasicDQN_CartPole` ``
- ``E`JuliaRL_DQN_CartPole` ``
- ``E`JuliaRL_PrioritizedDQN_CartPole` ``
- ``E`JuliaRL_Rainbow_CartPole` ``
- ``E`JuliaRL_IQN_CartPole` ``
- ``E`JuliaRL_A2C_CartPole` ``
- ``E`JuliaRL_A2CGAE_CartPole` ``
- ``E`JuliaRL_PPO_CartPole` ``

## Basic Components

Now let's take a closer look at what's in an experiment.

\output{./display_JuliaRL_BasicDQN_CartPole_2}

In the highest level, each experiment contains the following four parts:

- [Agent](#agent)
- [Environment](#environment)
- [Hook](#hook)
- [Stop Condition](#stop_condition)

\dfig{body;agent_env.png;The relation between **agent** and **env**. The agent takes in an environment and feed an action back. This process repeats until a stop condition meets. In each step, the agent needs to improve its policy in order to maximize the expected total reward.}

When executing ``run(E`JuliaRL_BasicDQN_CartPole`)``, it will be dispatched to `run(agent, env, stop_condition, hook)`. So it's just the same as running the following lines:

\aside{[Multiple Dispatch](https://docs.julialang.org/en/v1/manual/methods/) is fully utilized in this package. And it's the secret of high extensibility.}

```julia
experiment     = E`JuliaRL_BasicDQN_CartPole`
agent          = experiment.policy
env            = experiment.env
stop_condition = experiment.stop_condition
hook           = experiment.hook

run(agent, env, stop_condition, hook)
```

Now let's explain these components one by one.

### Stop Condition

A stop condition is used to determine when to stop an experiment. Two typical ones are [`StopAfterStep`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.StopAfterStep) and [`StopAfterEpisode`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.StopAfterEpisode). As you may have seen, the above experiment uses `StopAfterStep(10_000)` as the stop condition. Try to change the stop condition and see if it works as expected.

```julia
experiment = E`JuliaRL_BasicDQN_CartPole`
run(experiment.policy, experiment.env, StopAfterEpisode(100), experiment.hook)
```

At some point, you may need to learn [how write a customized stop condition](/guide/#how_to_write_a_customized_hook).

### Hook

The concept of hook in `ReinforcementLearning.jl` is mainly inspired by the
**two-way** callbacks in FastAI \dcite{howard2020fastai}:

> A callback should be available at every single point that code can be run
> during training, so that a user can customise every single detail of the
> training method;

> Every callback should be able to access every piece of information available
> at that stage in the training loop, including hyper-parameters, losses,
> gradients, input and target data, and so forth;

In fact, we extend the first kind of callback further in
`ReinforcementLearning.jl`. Thanks to multiple-dispatch in Julia, we can easily
customize the behavior of every detail in training, testing, evaluating stages.

You can check the list of provided hooks
[here](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#Hooks-1).
Two common hooks are
[`TotalRewardPerEpisode`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.TotalRewardPerEpisode)
and
[`StepsPerEpisode`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.StepsPerEpisode).

```julia:./ex1
using ReinforcementLearning # hide
using Plots
pyplot() #hide # hide

experiment = E`JuliaRL_BasicDQN_CartPole`
hook = TotalRewardPerEpisode()
run(experiment.policy, experiment.env, experiment.stop_condition, hook)
plot(hook.rewards)
savefig(joinpath(@OUTPUT, "episode.svg")) # hide
```

\dfig{body;episode.svg;Total reward of each episode during training.}

Still wondering how the tensorboard logging data is generated? Learn [how to use tensorboard](https://juliareinforcementlearning.org/guide/#how_to_use_tensorboard) and [how to write a customized hook](https://juliareinforcementlearning.org/guide/#how_to_write_a_customized_hook).

### Agent

An agent is an instance of `AbstractPolicy`. It is a functional object which
takes in an environment and returns an action.

```julia
action = agent(env)
```

In the above experiment, the `agent` is of type
[`Agent`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.Agent),
which is one of the most common policies in this package. We'll study how to
create, modify and update an agent in detail later. Suppose now we want to apply
another policy to the cart pole environment, a simple random policy. We can
simply replace the first argument with `RandomPolicy([1, 2])`. Here `[1,2]` are
valid actions to the `CartPoleEnv`.

\aside{Remember to install Plots with `] add Plots` first.}

```julia
using ReinforcementLearning

experiment = E`JuliaRL_BasicDQN_CartPole`

run(RandomPolicy([1,2]), experiment.env, experiment.stop_condition, experiment.hook)

println(experiment.description)
```

Just like what you did above, you can now watch the result based on the
description of the experiment.

### Environment

We've been using the `CartPoleEnv` for all the experiments above. But what does
it look like? By printing it in the REPL, we can see a lot of information about
it. Each of them are clearly described in [interface.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/blob/master/src/interface.jl).

```julia:./env_cart_pole
using ReinforcementLearning # hide
env = CartPoleEnv()
show(stdout, MIME"text/plain"(), env)  # hide
```

\output{./env_cart_pole}

Some people coming from the Python world may be familiar with the APIs defined
in **OpenAI/Gym**. Ours are very similar to them for simple environments:

```julia
reset!(env)              # reset env to the initial state
state(env)               # get the state from environment, usually it's a tensor
reward(env)              # get the reward since last interaction with environment
is_terminated(env)       # check if the game is terminated or not
actions(env)             # valid actions
env(rand(actions(env)))  # update the environment's internal state given an action
```

However, our package has a more ambitious goal to support much more complicated
environments. You may take a look at
[ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl)
to see some more built in examples. For users who are interested in applying
algorithms in this package to their own problems, you may also read the detailed description for [how to write a customized environment](http://juliareinforcementlearning.org/guide/#how_to_write_a_customized_environment).

## What's Next?

We have introduced the four main concepts in the `ReinforcementLearning.jl`
package. I hope you have a better understanding of them now.
- For starters who would like to learn reinforcement learning, I'd suggest you
  start from
  [ReinforcementLearningAnIntroduction.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl).
  If you are already familiar with traditional tabular reinforcement learning
  algorithms, then go ahead to
  [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl)
  to explore those DRL related
  [experiments](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/tree/master/src/experiments).
  Try to modify the parameters and compare the different results.
- For general users who want to use existing algorithms in our package to their
  customized environments, first learn skim through games defined in
  [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl)
  to learn how to describe the problem you are going to deal with. Then choose
  the appropriate policy in
  [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl)
  and tune the hyparameters. The [Guide](/guide) page may help you understand
  how each component is connected with others.
- For algorithm designers who want to contribute new algorithms, you're
  suggested to read the [blog](/blog) to understand the design principles and
  best practices.
