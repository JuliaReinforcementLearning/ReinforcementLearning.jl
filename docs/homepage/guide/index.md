@def title = "A Beginner's Guide to ReinforcementLearning.jl"
@def description = "From Novice to Professional"
@def is_enable_toc = true
@def bibliography = "bibliography.bib"

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
        "publishedDate":"2021-01-30",
        "citationText":"Jun Tian, 2021"
    }"""

Here we collect some common questions and answers to help you gain a better
understanding of `ReinforcementLearning.jl`.

## What are `legal_action_space` and `legal_action_space_mask`?

For environments of
[`FULL_ACTION_SET`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_base/#ReinforcementLearningBase.FULL_ACTION_SET),
the legal actions can not be determined ahead of time. So we need to define
`legal_action_space(env)` to return valid actions at each step. For environments
of
[MultiAgent](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_base/#ReinforcementLearningBase.MultiAgent-Tuple{Integer}),
`legal_action_space(env, player)` should also be defined. Also note that now the
result of `legal_action_space(env)` at each step must be a subset of
`action_space(env)`.

To handle the environments of `FULL_ACTION_SET` with discrete actions, some
algorithms need to know the mask of legal actions compared to the full actions
(the result of `action_space(env)`). For example, in neural network based
algorithms, we usually apply this mask to the last output layer to select legal
actions only. So the `legal_action_space_mask` may also be implemented in this
case. In most cases it can be simply defined like this:

```julia
RLBase.legal_action_space_mask(env::YourEnv) = map(action_space(env)) do action
    action in legal_action_space(env)
end
```

## How to write a customized environment?

See the detailed [blog](/blog/how_to_write_a_customized_environment/).

## How to write a environment wrapper?

Sometimes, you may want to write a new environment starting from existing
environments. To write a such environment wrapper, you only need to define your
structure as a subtype of `AbstractEnvWrapper` and store the original
environment in the `env` field. Then by default all environment related APIs
defined in `RLBase` will be forwarded into the inner `env`. You only need to
implement the interfaces as needed.

The following example defines a wrapper to clip the reward:

```julia
struct ClipRewardWrapper{T} <: AbstractEnvWrapper
    env::T
end

RLBase.reward(env::ClipRewardWrapper) = clamp(reward(env.env), -0.1, 0.1)
```

## How to write a customized stop condition?

Stop condition is just a function which is executed after interacting environment and returns a bool value indicating whether to stop an experiment or not.

```julia
function hook(agent, env)::Bool
    # ...
end
```

Usually a closure or a functional object will be used to store some intermediate data.

## How to write a customized hook?

In most cases, you don't need to write a customized hook. Some ver general hooks are provided so that you can inject any runtime logic at appropriate time:

- [`DoEveryNStep`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.DoEveryNStep)
- [`DoEveryNEpisode`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.DoEveryNEpisode)

However, if you do need to write a customized hook, the following methods must be provided:

- `(hook::YourHook)(::PreActStage, agent, env, action)`, note that there's an extra argument of `action`.
- `(hook::YourHook)(::PostActStage, agent, env)`
- `(hook::YourHook)(::PreEpisodeStage, agent, env)`
- `(hook::YourHook)(::PostEpisodeStage, agent, env)`

If your hook is a subtype of `AbstractHook`, then all the above methods will have a default implementation which just returns `nothing`. So that you only need to extend the necessary method you want.

## How to use TensorBoard?

This package adopts a non-invasive way for logging. So you can log everything you like with a hook. For example, to log the loss of each step. You can use the [`DoEveryNStep`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.DoEveryNStep).

```julia
DoEveryNStep() do t, agent, env
    with_logger(lg) do
        @info "training" loss = agent.policy.learner.loss
    end
end,
```

## How to evaluate an agent during training?

Well, just like the matryoshka doll, we run an experiment inside an experiment with a hook!

```julia
run(
    agent,
    env,
    stop_condition,
    DoEveryNStep(EVALUATION_FREQ) do t, agent, env
        run(agent, env, eval_stop_condition, eval_hook)
    end
)
```

\dfig{body;dolls.gif;From https://cdn.dribbble.com/users/882503/screenshots/3744602/dolls.gif}
