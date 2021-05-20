### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ dccad7c8-62fb-11eb-226f-393c73301bcb
begin
	using Dates
	using Pkg
end

# ╔═╡ 5938a9a6-6099-11eb-1ac9-1fb12d9c9237
using ReinforcementLearning

# ╔═╡ ad1302aa-609a-11eb-0a22-892b5c80750d
using Random

# ╔═╡ 06013422-609b-11eb-24e0-790eb0272183
using Plots

# ╔═╡ ac7e107c-609b-11eb-2d50-4b50517e1840
using Flux:InvDecay

# ╔═╡ bed48ef6-62fb-11eb-327a-ed29787dda3c
md"""
# How to Write a Customized Environment in ReinforcementLearning.jl?
"""

# ╔═╡ e262295a-62fb-11eb-32c0-051c8995d14a
md"""
- Last Update: $(now())
- Julia Version: $VERSION
- ReinforcementLearning.jl Version: $([v.version for (k,v) in Pkg.dependencies() if v.name=="ReinforcementLearning"][1])
"""

# ╔═╡ 03852aa2-6099-11eb-39db-ffba5eeade98
md"""
The first step to apply algorithms in ReinforcementLearning.jl is to define the problem you want to solve in a recognizable way. Here we'll demonstrate how to write many different kinds of environments based on interfaces defined in `ReinforcementLearningBase.jl`

The most commonly used interfaces to describe reinforcement learning tasks is [OpenAI/Gym](https://gym.openai.com/). Inspired by it, we expand those interfaces a little to utilize the multiple-dispatch in Julia and to cover multi-agent environments.

## The Minimal Interfaces to Implement

Many interfaces in [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl) have a default implementation. So in most cases, you only need to implement the following functions to define a customized environment:

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

Here we use an example introduced in [Monte Carlo Tree Search: A Tutorial](https://www.informs-sim.org/wsc18papers/includes/files/021.pdf) to demonstrate how to write a simple environment.

The game is defined like this: assume you have \$10 in your pocket, and you are faced with the following three choices:

1. Buy a PowerRich lottery ticket (win \$100M w.p. 0.01; nothing otherwise);
1. Buy a MegaHaul lottery ticket (win \$1M w.p. 0.05; nothing otherwise);
1. Do not buy a lottery ticket.

This game is a one-shot game. It terminates immediately after taking an action
and a reward is received. First we define a concrete subtype of `AbstractEnv`
named `LotteryEnv`:
"""

# ╔═╡ d4d6a2e0-6099-11eb-10ad-dd67c607ea0a
Base.@kwdef mutable struct LotteryEnv <: AbstractEnv
    reward::Union{Nothing, Int} = nothing
end

# ╔═╡ dee89ee4-6099-11eb-2c5a-9d05e6c4eb86
md"""
`LotteryEnv` has only one field named `reward`, by default it is initialized with `nothing`. Now let's implement the necessary interfaces:
"""

# ╔═╡ ef142626-6099-11eb-1d25-e1278a8c393e
RLBase.action_space(env::LotteryEnv) = (:PowerRich, :MegaHaul, nothing)

# ╔═╡ f02be148-6099-11eb-0b1b-95bd6aab04cb
md"""
Here `RLBase` is just an alias for `ReinforcementLearningBase`.
"""

# ╔═╡ 0f0ec27e-609a-11eb-3557-031e6004e78a
begin
	RLBase.reward(env::LotteryEnv) = env.reward
	RLBase.state(env::LotteryEnv) = !isnothing(env.reward)
	RLBase.state_space(env::LotteryEnv) = [false, true]
	RLBase.is_terminated(env::LotteryEnv) = !isnothing(env.reward)
	RLBase.reset!(env::LotteryEnv) = env.reward = nothing
end

# ╔═╡ 1ba95468-609a-11eb-1678-8bd0f3906606
md"""
Because the lottery game is just a simple one-shot game. If the `reward` is
`nothing` then the game is not started yet and we say the game is in state
`false`, otherwise the game is terminated and the state is `true`. So the result
of `state_space(env)` describes the possible states of this environment. By `reset!`
the game, we simply assign the reward with `nothing`, meaning that it's in the
initial state again.

The only left one is to implement the game logic:
"""

# ╔═╡ 52fd1580-609a-11eb-3de2-ed1e5ad95005
function (x::LotteryEnv)(action)
    if action == :PowerRich
        x.reward = rand() < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        x.reward = rand() < 0.05 ? 1_000_000 : -10
    elseif isnothing(action)
        x.reward = 0
    else
        @error "unknown action of $action"
    end
end

# ╔═╡ 57f86f42-609a-11eb-19af-2d1051ad81dd
md"""

## Test Your Environment

A method named `RLBase.test_runnable!` is provided to rollout several
simulations and see whether the environment we defined is functional.
"""

# ╔═╡ 5d8d1c48-609a-11eb-1fb4-a979b8aca3dc
env = LotteryEnv()

# ╔═╡ 6307f9e0-609a-11eb-1f92-c594d4aabeae
RLBase.test_runnable!(env)

# ╔═╡ 894620fa-609a-11eb-2665-93f5393ba35e
md"""
It is a simple smell test which works like this:

```
for _ in 1:n_episode
    reset!(env)
    while !is_terminated(env)
        env |> action_space |> rand |> env
    end
end
```
"""

# ╔═╡ 9a6f9636-609a-11eb-2357-9d10f0a52d7a
md"""
One step further is to test that other components in ReinforcementLearning.jl
also work. Similar to the test above, let's try the `RandomPolicy` first:
"""

# ╔═╡ b67979f0-609a-11eb-1efd-bb1407e6affc
run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000)) 

# ╔═╡ fc508c3e-609a-11eb-2b19-df3956c2fc7d
md"""
If no error shows up, then it means our environment at least works with the
`RandomPolicy` 🎉🎉🎉. Next, we can add a hook to collect the reward in each
episode to see the performance of the `RandomPolicy`.
"""

# ╔═╡ 04299018-609b-11eb-185e-83b941aa7462
begin
	hook = TotalRewardPerEpisode()
	run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000), hook)
	plot(hook.rewards)
end

# ╔═╡ a07eed5a-609b-11eb-2bbb-f318c9a796b5
md"""
A random policy is usually not very meaningful. Here we'll use a tabular based
monte carlo method to estimate the state-action value. (You may choose
appropriate algorithms based on the problem you're dealing with.)
"""

# ╔═╡ b443632a-609b-11eb-3976-5d199171c779
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

# ╔═╡ b9ebd5aa-609b-11eb-11cc-739730e6208e
p(env)

# ╔═╡ be7c6ac6-609b-11eb-2b32-8dd208db1c57
md"""
Oops, we get an error here. So what does it mean? 

Before answering this
question, let's spend some time on understanding the policy we defined above.
A
[`QBasedPolicy`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.QBasedPolicy)
contains two parts: a `learner` and an `explorer`. The `learner` *learn* the
state-action value function (aka *Q* function) duiring interactions with the
`env`. The `explorer` is used to select an action based on the Q value returned
by the `learner`. Here the [`EpsilonGreedyExplorer(0.1)`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.EpsilonGreedyExplorer) will select the action of the
largest value with probability `0.9` and select a random one with probability `0.1`. Inside of the `MonteCarloLearner`, a `TabularQApproximator` is
used to estimate the Q value.

That's the problem! A `TabularQApproximator` only
accepts states of type `Int`.
"""

# ╔═╡ e408057c-609b-11eb-27d5-ff2c082e98c6
p.learner.approximator(1, 1)  # Q(s, a)

# ╔═╡ ea06f580-609b-11eb-1828-ddc0e5744cc5
p.learner.approximator(1)     # [Q(s, a) for a in action_space(env)]

# ╔═╡ ed393fee-609b-11eb-1706-830287c2fbc7
p.learner.approximator(false)

# ╔═╡ f0eabad0-609b-11eb-0422-25937841f8e6
md"""
OK, now we know where the problem is. But how to fix it?

A initial idea is to
rewrite the `RLBase.state(env::LotteryEnv)` function to force it return an
`Int`. That's workable. But in some cases, we may be using environments written
by others and it's not very easy to modify the code directly. Fortunatelly, some
[built-in wrappers](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl/tree/master/src/environments/wrappers) are provided to help us transform the environment.
"""

# ╔═╡ 03eebcc6-609c-11eb-352c-398708e60ec3
wrapped_env = ActionTransformedEnv(
    StateOverriddenEnv(
        env,
        s -> s ? 1 : 2
    ),
    action_space_mapping = _ -> Base.OneTo(3),
    action_mapping = i -> action_space(env)[i]
)

# ╔═╡ 13535078-609c-11eb-0a3a-873d4d82a3b4
p(wrapped_env)

# ╔═╡ 3ad3704c-609c-11eb-15e5-0709c3246396
md"""
Nice job! Now we are ready to run the experiment:
"""

# ╔═╡ 4108c73c-609c-11eb-19ff-af4e1641eb8a
begin
	h = TotalRewardPerEpisode()
	run(p, wrapped_env, StopAfterEpisode(1_000), h)
	plot(h.rewards)
end

# ╔═╡ 59f80b90-609c-11eb-30ef-776f1154d897
md"""
If you are observant enough, you'll find that our policy is not updating at all!!!
"""

# ╔═╡ ab96c892-609c-11eb-08b4-e3f1d1d70049
p.learner.approximator.table

# ╔═╡ b63d5192-609c-11eb-1bdc-21dd1f56cf22
md"""
Well, actually the policy is running in the **evaluation** mode here. We'll explain it in another blog. For now, you only need to know that we can wrap the policy in an `Agent` to train the policy.
"""

# ╔═╡ c4e97848-609d-11eb-0e9d-dda4843007bd
agent = Agent(;
	policy=p,
	trajectory=VectorSARTTrajectory()
)

# ╔═╡ d6575e4c-609d-11eb-01be-63f485339089
new_hook = TotalRewardPerEpisode()

# ╔═╡ a4a1e976-609d-11eb-01aa-b32b0af8e34d
run(agent, wrapped_env, StopAfterStep(100_000), new_hook)	

# ╔═╡ 6c707936-609e-11eb-02dd-f14440c00c2f
p.learner.approximator.table  

# ╔═╡ 642b61fc-609c-11eb-23a4-25002f4a0bd3
md"""
!!! note
	Always remember that each algorithm usually only works in some specific environments, just like the `QBasedPolicy` above. So choose the right tool wisely 😉.
"""

# ╔═╡ c3b1dde0-60a6-11eb-214d-b7c966400eae
md"""
## More Complicated Environments

The above `LotteryEnv` is quite simple. Many environments we are interested in fall in the same category. Beyond that, there're still many other kinds of environments. You may take a glimpse at the [table](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl#built-in-environments) to see how many different types of environments are supported in [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl).

To distinguish different kinds of environments, some common traits are defined in [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl). Now we'll explain them one-by-one.

### `StateStyle`

In the above `LotteryEnv`, `state(env::LotteryEnv)` simply returns a `true` or `false`. But in some other environments, the function name `state` may be kind of vague. People with different background often talk about the same thing with different names. You may be interested in this discussion: [What is the difference between an observation and a state in reinforcement learning?](https://ai.stackexchange.com/questions/5970/what-is-the-difference-between-an-observation-and-a-state-in-reinforcement-learn) To avoid confusion when executing `state(env)`, the environment designer can explicitly define `state(::AbstractStateStyle, env::YourEnv)`. So that users can fetch necessary information on demand. Following are some built-in state styles:
"""

# ╔═╡ bd9cf900-60a9-11eb-1b11-8bc74ae5da60
subtypes(RLBase.AbstractStateStyle)

# ╔═╡ acf33c12-60af-11eb-00a8-b3e6a35dd256
md"""
Note that every state style may have different representations, `String`, `Array`, `Graph` and so on. All the above state styles can accept a data type as parameter. For example:
"""

# ╔═╡ 12fa94ec-60b0-11eb-0bd7-a5da4368db56
RLBase.state(::Observation{String}, env::LotteryEnv) = is_terminated(env) ? "Game Over" : "Game Start"

# ╔═╡ 43d13376-60b0-11eb-3f05-8f50e494764b
md"""
For environments which support many different kinds of states, developers should specify all the supported state styles. For example:
"""

# ╔═╡ e06eccf4-60b5-11eb-1511-31e0795dc87b
tp = TigerProblemEnv();

# ╔═╡ ccffa9cc-60b5-11eb-140c-676065b1c34f
StateStyle(tp)

# ╔═╡ e7564254-60b5-11eb-24e4-af25d66f7eba
state(tp, Observation{Int64}())

# ╔═╡ 4ef7543e-60b6-11eb-2883-735a555b0244
state(tp, InternalState{Int64}())

# ╔═╡ 56a92700-60b6-11eb-1af1-6500aa59fee2
state(tp)

# ╔═╡ 5cfd5da8-60b6-11eb-014d-e14fec02fd5d
DefaultStateStyle(tp)

# ╔═╡ 78fddc1c-60b6-11eb-1266-75360b57e742
md"""
### `DefaultStateStyle`

The `DefaultStateStyle` trait returns the first element in the result of `StateStyle` by default.

For algorithm developers, they usually don't care about the state style. They can assume that the default state style is always well defined and simply call `state(env)` to get the right representation. So for environments of many different representations, `state(env)` will be dispatched to `state(DefaultStateStyle(env), env)`. And we can use the [`DefaultStateStyleEnv`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.DefaultStateStyleEnv-Union{Tuple{E},%20Tuple{S}}%20where%20E%20where%20S) wrapper to override the pre-defined `DefaultStateStyle(::YourEnv)`.
"""

# ╔═╡ 8ddb2a40-60b6-11eb-1396-d9880e304217
md"""
### `RewardStyle`

For games like Chess, Go or many card game, we only get the reward at the end of an game. We say this kind of games is of `TerminalReward`, otherwise we define it as `StepReward`. Actually the `TerminalReward` is a special case of `StepReward` (for non-terminal steps, the reward is `0`). The reason we still want to distinguish these two cases is that, for some algorithms there may be a more efficient implementation for `TerminalReward` style games.

"""

# ╔═╡ 97cba458-60b6-11eb-2b9e-b5ca1f9a5c73
RewardStyle(tp)

# ╔═╡ b2dab6da-60b6-11eb-275b-57d0a3e65da8
RewardStyle(MontyHallEnv())

# ╔═╡ b8e097f8-60b5-11eb-27ab-7f73e2d7c502
md"""
### `ActionStyle`

For some environments, the valid actions in each step may be different. We call this kind of environments are of `FullActionSet`. Otherwise, we say the environment is of `MinimalActionSet`. A typical built-in environment with `FullActionSet` is the `TicTacToeEnv`. Two extra methods must be implemented:
"""

# ╔═╡ 81614ebc-60b5-11eb-15b9-897cdcbde04f
ttt = TicTacToeEnv();

# ╔═╡ a1c73c42-60b3-11eb-3db7-d71785bb4370
ActionStyle(ttt)

# ╔═╡ 6127b88e-60b5-11eb-2bbc-4d078ce6b44e
legal_action_space(ttt)

# ╔═╡ 8a0e6db0-60b5-11eb-3263-295fc6a951a2
legal_action_space_mask(ttt)

# ╔═╡ 9558df16-60b5-11eb-3fbd-77c38bbe6ca4
md"""
### `NumAgentStyle`

In the above `LotteryEnv`, only one player is involved in the environment. In many board games, usually multiple players are engaged.
"""


# ╔═╡ 4060756c-60b7-11eb-3e2f-a59544eae5c8
NumAgentStyle(env)

# ╔═╡ 4d36f23e-60b7-11eb-39be-494f4cb0fa24
NumAgentStyle(ttt)

# ╔═╡ 566c4a16-60b7-11eb-2ced-0b04dbc2953d
md"""
For multi-agent environments, some new APIs are introduced. The meaning of some APIs we've seen are also extended.

First, multi-agent environment developers must implement `players` to distinguish different players.
"""

# ╔═╡ 17cbc880-60b8-11eb-0b27-d3cbdadb4093
players(ttt)

# ╔═╡ 4ade9f26-60b9-11eb-3a64-497b0ab2c25f
current_player(ttt)

# ╔═╡ 34e6633a-60b8-11eb-375f-cd0093097caa
md"""

| Single Agent | Multi-Agent |
| ------------:| -----------:|
| `state(env)` | `state(env, player)`|
| `reward(env)`| `reward(env, player)`|
| `env(action)`| `env(action, player)`|
| `action_space(env)`| `action_space(env, player)`|
| `state_space(env)`| `state_space(env, player)`|
| `is_terminated(env)` | `is_terminated(env, player)`|

Note that the APIs in single agent is still valid, only that they all fall back to the perspective from the `current_player(env)`.
"""

# ╔═╡ 61f6fd5c-60b9-11eb-1062-f1aff5da84af
md"""
#### `UtilityStyle`

In multi-agent environments, sometimes the sum of rewards from all players are always `0`. We call the `UtilityStyle` of these environments `ZeroSum`. `ZeroSum` is a special case of `ConstantSum`. In cooperational games, the reward of each player are the same. In this case, they are called `IdenticalUtility`. Other cases fall back to `GeneralSum`.
"""

# ╔═╡ 44cad7c0-60ba-11eb-2570-b1da3d16c973
md"""
#### `InformationStyle`

If all players can see the same state, then we say the `InformationStyle` of these environments are of `PerfectInformation`. They are a special case of `ImperfectInformation` environments.
"""

# ╔═╡ 80f3f7a4-60ba-11eb-301f-71b73ece720f
md"""
#### `DynamicStyle`

All the environments we've seen so far were of `Sequential` style, meaning that at each step, only **ONE** player was allowed to take an action. Alternatively there are `Simultaneous` environments, where all the players take actions simultaneously without seeing each other's action in advance. Simultaneous environments must take a collection of actions from different players as input.
"""

# ╔═╡ 6a58f464-60bb-11eb-12ab-592202a61d9f
rps = RockPaperScissorsEnv();

# ╔═╡ 5cf93fca-60bb-11eb-1376-37b9322258d4
action_space(rps)

# ╔═╡ 668165fe-60bb-11eb-0b0b-bf0fc2f2e36a
rps(rand(action_space(rps)))

# ╔═╡ 7e056cde-60bb-11eb-17e6-3bcefeeae0b0
md"""
#### `ChanceStyle`

If there's no `rng` in the environment, everything is deterministic afer taking each action, then we call the `ChanceStyle` of these environments are of `Deterministic`. Otherwise, we call them `Stochastic`. One special case is that, in [Extensive Form Games](https://en.wikipedia.org/wiki/Extensive-form_game), a chance node is envolved. And the action probability of this special player is known. For these environments, we need to have the following methods defined:
"""

# ╔═╡ 2b203eb2-60bc-11eb-30e6-9beaf61fe7a9
kp = KuhnPokerEnv();

# ╔═╡ 48707c20-60bc-11eb-2ac4-0da804dd9381
chance_player(kp)

# ╔═╡ 426a9bb2-60bc-11eb-1efd-f385e77e9c22
prob(kp, chance_player(kp))

# ╔═╡ 51cdadce-60bc-11eb-01f7-e1e9b6362ad3
chance_player(kp) in players(kp)

# ╔═╡ 6b9e0564-60bc-11eb-196e-09b9b8f557ba
md"""
## Examples

Finally we've gone through all the details you need to know for how to write a customized environment. You're encouraged to take a look at the examples provided in [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl). Feel free to create an issue there if you're still not sure how to describe your problem with the interfaces defined in this package.
"""

# ╔═╡ Cell order:
# ╟─bed48ef6-62fb-11eb-327a-ed29787dda3c
# ╟─dccad7c8-62fb-11eb-226f-393c73301bcb
# ╟─e262295a-62fb-11eb-32c0-051c8995d14a
# ╟─03852aa2-6099-11eb-39db-ffba5eeade98
# ╠═5938a9a6-6099-11eb-1ac9-1fb12d9c9237
# ╠═d4d6a2e0-6099-11eb-10ad-dd67c607ea0a
# ╟─dee89ee4-6099-11eb-2c5a-9d05e6c4eb86
# ╠═ef142626-6099-11eb-1d25-e1278a8c393e
# ╟─f02be148-6099-11eb-0b1b-95bd6aab04cb
# ╠═0f0ec27e-609a-11eb-3557-031e6004e78a
# ╟─1ba95468-609a-11eb-1678-8bd0f3906606
# ╠═52fd1580-609a-11eb-3de2-ed1e5ad95005
# ╟─57f86f42-609a-11eb-19af-2d1051ad81dd
# ╠═5d8d1c48-609a-11eb-1fb4-a979b8aca3dc
# ╠═6307f9e0-609a-11eb-1f92-c594d4aabeae
# ╟─894620fa-609a-11eb-2665-93f5393ba35e
# ╟─9a6f9636-609a-11eb-2357-9d10f0a52d7a
# ╠═ad1302aa-609a-11eb-0a22-892b5c80750d
# ╠═b67979f0-609a-11eb-1efd-bb1407e6affc
# ╟─fc508c3e-609a-11eb-2b19-df3956c2fc7d
# ╠═06013422-609b-11eb-24e0-790eb0272183
# ╠═04299018-609b-11eb-185e-83b941aa7462
# ╟─a07eed5a-609b-11eb-2bbb-f318c9a796b5
# ╠═ac7e107c-609b-11eb-2d50-4b50517e1840
# ╠═b443632a-609b-11eb-3976-5d199171c779
# ╠═b9ebd5aa-609b-11eb-11cc-739730e6208e
# ╟─be7c6ac6-609b-11eb-2b32-8dd208db1c57
# ╠═e408057c-609b-11eb-27d5-ff2c082e98c6
# ╠═ea06f580-609b-11eb-1828-ddc0e5744cc5
# ╠═ed393fee-609b-11eb-1706-830287c2fbc7
# ╟─f0eabad0-609b-11eb-0422-25937841f8e6
# ╠═03eebcc6-609c-11eb-352c-398708e60ec3
# ╠═13535078-609c-11eb-0a3a-873d4d82a3b4
# ╟─3ad3704c-609c-11eb-15e5-0709c3246396
# ╠═4108c73c-609c-11eb-19ff-af4e1641eb8a
# ╟─59f80b90-609c-11eb-30ef-776f1154d897
# ╠═ab96c892-609c-11eb-08b4-e3f1d1d70049
# ╟─b63d5192-609c-11eb-1bdc-21dd1f56cf22
# ╠═c4e97848-609d-11eb-0e9d-dda4843007bd
# ╠═d6575e4c-609d-11eb-01be-63f485339089
# ╠═a4a1e976-609d-11eb-01aa-b32b0af8e34d
# ╠═6c707936-609e-11eb-02dd-f14440c00c2f
# ╟─642b61fc-609c-11eb-23a4-25002f4a0bd3
# ╟─c3b1dde0-60a6-11eb-214d-b7c966400eae
# ╠═bd9cf900-60a9-11eb-1b11-8bc74ae5da60
# ╟─acf33c12-60af-11eb-00a8-b3e6a35dd256
# ╠═12fa94ec-60b0-11eb-0bd7-a5da4368db56
# ╟─43d13376-60b0-11eb-3f05-8f50e494764b
# ╠═e06eccf4-60b5-11eb-1511-31e0795dc87b
# ╠═ccffa9cc-60b5-11eb-140c-676065b1c34f
# ╠═e7564254-60b5-11eb-24e4-af25d66f7eba
# ╠═4ef7543e-60b6-11eb-2883-735a555b0244
# ╠═56a92700-60b6-11eb-1af1-6500aa59fee2
# ╠═5cfd5da8-60b6-11eb-014d-e14fec02fd5d
# ╟─78fddc1c-60b6-11eb-1266-75360b57e742
# ╟─8ddb2a40-60b6-11eb-1396-d9880e304217
# ╠═97cba458-60b6-11eb-2b9e-b5ca1f9a5c73
# ╠═b2dab6da-60b6-11eb-275b-57d0a3e65da8
# ╟─b8e097f8-60b5-11eb-27ab-7f73e2d7c502
# ╠═81614ebc-60b5-11eb-15b9-897cdcbde04f
# ╠═a1c73c42-60b3-11eb-3db7-d71785bb4370
# ╠═6127b88e-60b5-11eb-2bbc-4d078ce6b44e
# ╠═8a0e6db0-60b5-11eb-3263-295fc6a951a2
# ╟─9558df16-60b5-11eb-3fbd-77c38bbe6ca4
# ╠═4060756c-60b7-11eb-3e2f-a59544eae5c8
# ╠═4d36f23e-60b7-11eb-39be-494f4cb0fa24
# ╟─566c4a16-60b7-11eb-2ced-0b04dbc2953d
# ╠═17cbc880-60b8-11eb-0b27-d3cbdadb4093
# ╠═4ade9f26-60b9-11eb-3a64-497b0ab2c25f
# ╟─34e6633a-60b8-11eb-375f-cd0093097caa
# ╟─61f6fd5c-60b9-11eb-1062-f1aff5da84af
# ╟─44cad7c0-60ba-11eb-2570-b1da3d16c973
# ╟─80f3f7a4-60ba-11eb-301f-71b73ece720f
# ╠═6a58f464-60bb-11eb-12ab-592202a61d9f
# ╠═5cf93fca-60bb-11eb-1376-37b9322258d4
# ╠═668165fe-60bb-11eb-0b0b-bf0fc2f2e36a
# ╟─7e056cde-60bb-11eb-17e6-3bcefeeae0b0
# ╠═2b203eb2-60bc-11eb-30e6-9beaf61fe7a9
# ╠═48707c20-60bc-11eb-2ac4-0da804dd9381
# ╠═426a9bb2-60bc-11eb-1efd-f385e77e9c22
# ╠═51cdadce-60bc-11eb-01f7-e1e9b6362ad3
# ╟─6b9e0564-60bc-11eb-196e-09b9b8f557ba
