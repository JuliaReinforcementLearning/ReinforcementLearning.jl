# Episodic vs Non-episodic environments

## Episodic environments
By default, `run(policy, env, stop_condition, hook)` will step through `env` until a terminal state is reached, signaling the end of an episode. To be able to do so, `env` must implement the `RLBase.is_terminated(::YourEnvironment)` function. This function is called after each step through the environment and when it returns `true`, the trajectory records the terminal state, then the `RLBase.reset!(::YourEnvironment)` function is called and the environment is set to (one of) its initial state(s). 

Using this means that the value of the terminal state is set to 0 when learning its value via boostrapping.

## Non-episodic environment

Also called _Continuing tasks_ (Sutton & Barto, 2018), non-episodic environment do not have a terminal state and thus may run for ever, or until the `stop_condition` is reached. Sometimes however, one may want to periodically reset the environment to start fresh. A first possibility is to implement `RLBase.is_terminated(::YourEnvironment)` to reset according to an arbitrary condition. However this may not be a good idea because the value of the last state (note that it is not a _terminal_ state) will be bootstrapped to 0 during learning, even though it is not the true value of the state. 

To manage this, we provide the `ResetAfterNSteps(n)` condition as an argument to `run(policy, env, stop_condition, hook, reset_condition = ResetIfEnvTerminated())`. The default `ResetIfEnvTerminated()` assumes an episodic environment, changing that to `ResetAfterNSteps(n)` will no longer check `is_terminated` but will instead call `reset!` every `n` steps. This way, the value of the last state will not be multiplied by 0 during bootstrapping and the correct value can be learned. 

## Custom reset conditions

You can specify a custom `reset_condition` instead of using the built-in's. Your condition must be callable with the method `RLCore.check!(my_condition, policy, env)`. For example, here is how to implement a custom condition that checks for a terminal state but will also reset if the episode is too long:

```julia
using ReinforcementLearning
import ReinforcementLearning: RLCore
reset_n_steps = ResetAfterNSteps(10000)

struct MyCondition <: AbstractResetCondition end

function RLCore.check!(my_condition::MyCondition, policy, env)
    terminal = is_terminated(env)
    too_long = RLCore.check!(reset_n_steps, policy, env)
    return terminal || too_long
end
env = RandomWalk1D()
agent = RandomPolicy()
stop_condition = StopIfEnvTerminated()
hook = EmptyHook()
run(agent, env, stop_condition, hook, MyCondition())
```

We can instead make a callable struct instead of a function to avoid the global `reset_n_step`. 

```julia
mutable struct MyCondition1 <: AbstractResetCondition
    reset_after
end

RLCore.check!(c::MyCondition1, policy, env) = is_terminated(env) || RLCore.check!(c.reset_after, policy, env)

run(agent, env, stop_condition, hook, MyCondition1(ResetAfterNSteps(10000)))
```
