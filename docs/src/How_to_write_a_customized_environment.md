# How to write a customized environment?

The first step to apply algorithms in ReinforcementLearning.jl is to define the
problem you want to solve in a recognizable way. Here we'll demonstrate how to
write many different kinds of environments based on interfaces defined in
[ReinforcementLearningBase.jl](@ref).

The most commonly used interfaces to describe reinforcement learning tasks is
[OpenAI/Gym](https://gym.openai.com/). Inspired by it, we expand those
interfaces a little to utilize the multiple-dispatch in Julia and to cover
multi-agent environments.

## The Minimal Interfaces to Implement

Many interfaces in
[ReinforcementLearningBase.jl](@ref)
have a default implementation. So in most cases, you only need to implement the
following functions to define a customized environment:

```julia
action_space(env::YourEnv)
state(env::YourEnv)
state_space(env::YourEnv)
reward(env::YourEnv)
is_terminated(env::YourEnv)
reset!(env::YourEnv)
(env::YourEnv)(action)
```

## An Example: The LotteryEnv

Here we use an example introduced in [Monte Carlo Tree Search: A
Tutorial](https://www.informs-sim.org/wsc18papers/includes/files/021.pdf) to
demonstrate how to write a simple environment.

The game is defined like this: assume you have \$10 in your pocket, and you are
faced with the following three choices:

1. Buy a PowerRich lottery ticket (win \$100M w.p. 0.01; nothing otherwise);
1. Buy a MegaHaul lottery ticket (win \$1M w.p. 0.05; nothing otherwise);
1. Do not buy a lottery ticket.

This game is a one-shot game. It terminates immediately after taking an action
and a reward is received. First we define a concrete subtype of `AbstractEnv`
named `LotteryEnv`:

```@repl customized_env
using ReinforcementLearning
Base.@kwdef mutable struct LotteryEnv <: AbstractEnv
    reward::Union{Nothing, Int} = nothing
end
```

The `LotteryEnv` has only one field named `reward`, by default it is
initialized with `nothing`. Now let's implement the necessary interfaces:

```@repl customized_env
RLBase.action_space(env::LotteryEnv) = (:PowerRich, :MegaHaul, nothing)
```

Here `RLBase` is just an alias for `ReinforcementLearningBase`.

```@repl customized_env
RLBase.reward(env::LotteryEnv) = env.reward
RLBase.state(env::LotteryEnv) = !isnothing(env.reward)
RLBase.state_space(env::LotteryEnv) = [false, true]
RLBase.is_terminated(env::LotteryEnv) = !isnothing(env.reward)
RLBase.reset!(env::LotteryEnv) = env.reward = nothing
```

Because the lottery game is just a simple one-shot game. If the `reward`
is `nothing` then the game is not started yet and we say the game is in state
`false`, otherwise the game is terminated and the state is `true`. So the result
of `state_space(env)` describes the possible states of this environment. By
`reset!` the game, we simply assign the reward with `nothing`, meaning that it's
in the initial state again.

The only left one is to implement the game logic:

```@repl customized_env
function (x::LotteryEnv)(action)
    if action == :PowerRich
        x.reward = rand() < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        x.reward = rand() < 0.05 ? 1_000_000 : -10
    elseif isnothing(action) x.reward = 0
    else
        @error "unknown action of $action"
    end
end
```

## Test Your Environment

A method named `RLBase.test_runnable!` is provided to rollout several
simulations and see whether the environment we defined is functional.

```@repl customized_env
env = LotteryEnv()
RLBase.test_runnable!(env)
```

It is a simple smell test which works like this:

```
for _ in 1:n_episode
    reset!(env)
    while !is_terminated(env)
        env |> action_space |> rand |> env
    end
end
```

One step further is to test that other components in
ReinforcementLearning.jl also work. Similar to the test above, let's try the
[`RandomPolicy`](@ref) first:

```@repl customized_env
run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000)) 
```

If no error shows up, then it means our environment at least works with
the [`RandomPolicy`](@ref) 🎉🎉🎉. Next, we can add a hook to collect the reward in each
episode to see the performance of the `RandomPolicy`.

```@repl customized_env
hook = TotalRewardPerEpisode()
run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000), hook)
using Plots
pyplot() #hide
plot(hook.rewards)
savefig("custom_env_random_policy_reward.svg"); nothing # hide
```

![](custom_env_random_policy_reward.svg)

## Add an Environment Wrapper

Now suppose we'd like to use a tabular based monte carlo method to estimate the
state-action value.

```@repl customized_env
using Flux: InvDecay
p = QBasedPolicy(
    learner = MonteCarloLearner(;
        approximator=TabularQApproximator(
            ;n_state = length(state_space(env)),
            n_action = length(action_space(env)),
            opt = InvDecay(1.0)
        )
    ),
    explorer = EpsilonGreedyExplorer(0.1)
)
p(env)
```

Oops, we get an error here. So what does it mean? 

Before answering this question, let's spend some time on understanding the
policy we defined above. A [`QBasedPolicy`](@ref)
contains two parts: a `learner` and an `explorer`. The `learner` *learn* the
state-action value function (aka *Q* function) during interactions with the
`env`. The `explorer` is used to select an action based on the Q value returned
by the `learner`. Inside of the [`MonteCarloLearner`](@ref), a
[`TabularQApproximator`](@ref) is used to estimate the Q value.

That's the problem! A [`TabularQApproximator`](@ref) only accepts states of type `Int`.

```@repl customized_env
p.learner.approximator(1, 1)  # Q(s, a)
p.learner.approximator(1)     # [Q(s, a) for a in action_space(env)]
p.learner.approximator(false)
```

OK, now we know where the problem is. But how to fix it?

An initial idea is to rewrite the `RLBase.state(env::LotteryEnv)` function to
force it return an `Int`. That's workable. But in some cases, we may be using
environments written by others and it's not very easy to modify the code
directly. Fortunatelly, some environment wrappers are provided to help us
transform the environment.

```@repl customized_env
wrapped_env = ActionTransformedEnv(
    StateTransformedEnv(
        env;
        state_mapping=s -> s ? 1 : 2,
        state_space_mapping = _ -> Base.OneTo(2)
    );
    action_mapping = i -> action_space(env)[i],
    action_space_mapping = _ -> Base.OneTo(3),
)
p(wrapped_env)
```

Nice job! Now we are ready to run the experiment:

```@repl customized_env
h = TotalRewardPerEpisode()
run(p, wrapped_env, StopAfterEpisode(1_000), h)
plot(h.rewards)
savefig("custom_env_random_policy_reward_wrapped_env.svg"); nothing # hide
```

![](custom_env_random_policy_reward_wrapped_env.svg)


!!! warning
    If you are observant enough, you'll find that our policy is not updating
    at all!!! Actually, it's running in the **actor** mode. To update the policy,
    remember to wrap it in an [`Agent`](@ref).

## More Complicated Environments

The above `LotteryEnv` is quite simple. Many environments we are interested in
fall in the same category. Beyond that, there're still many other kinds of
environments. You may take a glimpse at the [Built-in Environments](@ref)
to see how many different types of environments are supported.

To distinguish different kinds of environments, some common traits are defined
in [ReinforcementLearningBase.jl](@ref). Now let's explain them one-by-one.

### [`StateStyle`](@ref)

In the above `LotteryEnv`, `state(env::LotteryEnv)` simply returns a boolean.
But in some other environments, the function name `state` may be kind
of vague. People from different background often talk about the same thing with
different names. You may be interested in this discussion: [What is the
difference between an observation and a state in reinforcement
learning?](https://ai.stackexchange.com/questions/5970/what-is-the-difference-between-an-observation-and-a-state-in-reinforcement-learn)
To avoid confusion when executing `state(env)`, the environment designer can
explicitly define `state(::AbstractStateStyle, env::YourEnv)`. So that users can
fetch necessary information on demand. Following are some built-in state styles:

```@repl customized_env
using InteractiveUtils
subtypes(RLBase.AbstractStateStyle)
```

Note that every state style may have many different representations, `String`,
`Array`, `Graph` and so on. All the above state styles can accept a data type as
parameter. For example:

```@repl customized_env
RLBase.state(::Observation{String}, env::LotteryEnv) = is_terminated(env) ? "Game Over" : "Game Start"
```

For environments which support many different kinds of states, developers
should specify all the supported state styles. For example:

```@repl customized_env
tp = TigerProblemEnv();
StateStyle(tp)
state(tp, Observation{Int64}())
state(tp, InternalState{Int64}())
state(tp)
```

### [`DefaultStateStyle`](@ref)

The [`DefaultStateStyle`](@ref) trait returns the first element in the result of
[`StateStyle`](@ref) by default.

For algorithm developers, they usually don't care about the state style. They
can assume that the default state style is always well defined and simply call
`state(env)` to get the right representation. So for environments of many
different representations, `state(env)` will be dispatched to
`state(DefaultStateStyle(env), env)`. And we can use the
[`DefaultStateStyleEnv`](@ref) wrapper to override the pre-defined `DefaultStateStyle(::YourEnv)`.

### [`RewardStyle`](@ref)

For games like Chess, Go or many card game, we only get the reward at the end of
an game. We say this kind of games is of [`TerminalReward`](@ref), otherwise we define
it as [`StepReward`](@ref). Actually the `TerminalReward` is a special case of
`StepReward` (for non-terminal steps, the reward is `0`). The reason we still
want to distinguish these two cases is that, for some algorithms there may be a
more efficient implementation for `TerminalReward` style games.

```@repl customized_env
RewardStyle(tp)
RewardStyle(MontyHallEnv())
```

### [`ActionStyle`](@ref)

For some environments, the valid actions in each step may be different. We call
this kind of environments are of [`FullActionSet`](@ref). Otherwise, we say the
environment is of [`MinimalActionSet`](@ref). A typical built-in environment with
[`FullActionSet`](@ref) is the [`TicTacToeEnv`](@ref). Two extra methods must be implemented:

```@repl customized_env
ttt = TicTacToeEnv();
ActionStyle(ttt)
legal_action_space(ttt)
legal_action_space_mask(ttt)
```

For some simple environments, we can simply use a `Tuple` or a `Vector` to
describe the action space. A special space type [`Space`](@ref) is also provided
as a meta space to hold the composition of different kinds of sub-spaces. For
example, we can use `Space(((1:3),(true,false)))` to describe the environment
with two kinds of actions, an integer between `1` and `3`, and a boolearn.
Sometimes, the action space is not easy to be described by some built in data
structures. In that case, you can defined a customized one with the following
interfaces implemented:

- `Base.in`
- `Random.rand`

For example, to define an action space on the N dimensional simplex:

```@repl customized_env
using Random

struct SimplexSpace
    n::Int
end

function Base.in(x::AbstractVector, s::SimplexSpace)
    length(x) == s.n && all(>=(0), x) && isapprox(1, sum(x))
end

function Random.rand(rng::AbstractRNG, s::SimplexSpace)
    x = rand(rng, s.n)
    x ./= sum(x)
    x
end
```

### [`NumAgentStyle`](@ref)

In the above `LotteryEnv`, only one player is involved in the environment. In
many board games, usually multiple players are engaged.

```@repl customized_env
NumAgentStyle(env)
NumAgentStyle(ttt)
```

For multi-agent environments, some new APIs are introduced. The meaning of
some APIs we've seen are also extended. First, multi-agent environment developers must implement `players` to
distinguish different players.

```@repl customized_env
players(ttt)
current_player(ttt)
```

| Single Agent | Multi-Agent |
| ------------:| -----------:|
| `state(env)` | `state(env, player)`|
| `reward(env)`| `reward(env, player)`|
| `env(action)`| `env(action, player)`|
| `action_space(env)`| `action_space(env, player)`|
| `state_space(env)`| `state_space(env, player)`|
| `is_terminated(env)` | `is_terminated(env, player)`|

Note that the APIs in single agent is still valid, only that they all fall back
to the perspective from the `current_player(env)`.

### [`UtilityStyle`](@ref)

In multi-agent environments, sometimes the sum of rewards from all players are
always `0`. We call the [`UtilityStyle`](@ref) of these environments [`ZeroSum`](@ref).
`ZeroSum` is a special case of [`ConstantSum`](@ref). In cooperational games, the reward
of each player are the same. In this case, they are called [`IdenticalUtility`](@ref).
Other cases fall back to [`GeneralSum`](@ref).

### [`InformationStyle`](@ref)

If all players can see the same state, then we say the [`InformationStyle`](@ref) of
these environments are of [`PerfectInformation`](@ref). They are a special case of
[`ImperfectInformation`](@ref) environments.

### [`DynamicStyle`](@ref)

All the environments we've seen so far were of [`Sequential`](@ref) style, meaning that
at each step, only **ONE** player was allowed to take an action. Alternatively
there are [`Simultaneous`](@ref) environments, where all the players take actions
simultaneously without seeing each other's action in advance. Simultaneous
environments must take a collection of actions from different players as input.

```@repl customized_env
rps = RockPaperScissorsEnv();
action_space(rps)
rps(rand(action_space(rps)))
```

### [`ChanceStyle`](@ref)

If there's no `rng` in the environment, everything is deterministic after taking
each action, then we call the [`ChanceStyle`](@ref) of these environments are of
[`Deterministic`](@ref). Otherwise, we call them [`Stochastic`](@ref), which is the
default return value. One special case is that,
in [Extensive Form Games](https://en.wikipedia.org/wiki/Extensive-form_game), a
chance node is involved. And the action probability of this special player is
determined. We define the `ChanceStyle` of these environments as [`EXPLICIT_STOCHASTIC`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.EXPLICIT_STOCHASTIC).
For these environments, we need to have the following methods defined:

```@repl customized_env
kp = KuhnPokerEnv();
chance_player(kp)
prob(kp, chance_player(kp))
chance_player(kp) in players(kp)
```

To explicitly specify the chance style of your custom environment, you can provide a specific dispatch of [`ChanceStyle`](@ref) for your custom environment.

## Examples

Finally we've gone through all the details you need to know for how to write a
customized environment. You're encouraged to take a look at the examples
provided in
[ReinforcementLearningEnvironments.jl](@ref).
Feel free to create an issue there if you're still not sure how to describe your
problem with the interfaces defined in this package.
