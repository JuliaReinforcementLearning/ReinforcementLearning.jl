# Maximum a Posterio Policy Optimization

ReinforcementLearningZoo proposes an implementation of the Maximum a Posterio Policy Optimization (MPO) algorithm. This algorithm was initially proposed by [Abdolmaleki et al. (2018)](https://arxiv.org/abs/1806.06920) and is further detailled in a [subsequent paper](https://arxiv.org/abs/1812.02256). This implementation is not identical to that of the paper for several reasons that we will detail later. The purpose of this page is to guide a RLZoo user through the creation of an experiment that uses the MPO algorithm. We will recreate [one of the three experiments](../../../src/ReinforcementLearningExperiments/deps/experiments/experiments/Policy%20Gradient/JuliaRL_MPO_CartPole.jl) available in RLExperiments.jl. 

The implementation of MPO is declined in three forms (one for each cartpole experiments):
  
- With a Categorical Actor (for discrete action spaces)
- With a Diagonal Gaussian (the standard actor for continuous action spaces in RL)
- With a Full Gaussian (which can learn a covariance between the different action dimensions)

The latter is the approach used in the paper for continuous actions. It is implemented but is very slow on a GPU at the moment. Although more expressive, it may not be worth the extra computation time. 

## Learning a continuous Cartpole policy
First, we instantiate the environment from the package `ReinforcementLearningEnvironments`. We wrap it into an `ActionTransformedEnv` with a `tanh` to constrain the action in [-1, 1].

```julia
using ReinforcementLearning, Flux

env = ActionTransformedEnv(CartPoleEnv(continuous = true), action_mapping = x->tanh(only(x)))
```

Because we want our experiment to be reproducible, we also use a seed.

```julia
using Random
Random.set_global_seed!(123)
```

Then we instantiate a `MPOPolicy` 
```julia
    policy = MPOPolicy(
        actor = Approximator(GaussianNetwork(
            Chain(Dense(4, 64, tanh), Dense(64,64,tanh)),
            Dense(64, 1),
            Dense(64, 1)), Adam(3f-4)),
        qnetwork1 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,64,gelu), Dense(64,1)), Adam(3f-4)),
        qnetwork2 = Approximator(Chain(Dense(5, 64, gelu), Dense(64,64,gelu), Dense(64,1)), Adam(3f-4)),
        action_sample_size = 32,
        ϵμ = 0.1f0, 
        ϵΣ = 1f-2,
        ϵ = 0.1f0)
```
`MPOPolicy` needs an Actor that is an `Approximator`, we use a Deep Neural Network and the `Adam` Optimiser from the `Flux.jl` package. Notice that the NN is a `GaussianNetwork` made of three parts. The first is a common body with an input size equal to the length of the state of the environment (4 in this case). Then we have two "heads", one for the mean of the Gaussian policy, and one for the standard deviation. Both heads must have the same output size (the size of the action vectors, 1 in this case) with a `GaussianNetwork` and no activation at the output layers. In

Then we have `qnetwork1` and 2. This implementation of MPO uses twin QNetworks with targets. Both must be `Approximator`s, but must not necessarily have the same architecture. The input size should be the size of the state + the size of the action (5). The output size must be 1. The original MPO paper uses the Retrace algorithm instead of 1-step TD to train the critics. This currently not implemented in RL.jl.

`MPOPolicy` has several keyword arguments in its constructor. We omit the least important ones here (that are not specific to MPO). You can see them using `?MPOPolicy` in the REPL. 

- `action_sample_size` is the number of actions sampled for each state during the E-step of the algorithm ($K$ in the second paper). 
- `ϵ` is the maximum KL divergence between the E-step variational distribution and the current policy.
- `ϵμ` is the maximum KL divergence between the updated policy at the M-step and the current policy, with respect to the mean of the Gaussian.
-  `ϵΣ` is the maximum KL divergence between the updated policy at the M-step and the current policy, with respect to the standard deviation of the Gaussian. It should typically be lower than `ϵμ` to ensure it does not shrink to 0 before the mean settles around its optimum. 
- `α_scale = 1f0` and `αΣ_scale = 100f0`, are the gradient descent learning rate for the lagrange penalty for the mean and covariance. We leave it to the default values here. 

The next step is to wrap this policy into an `Agent`. An agent is a combination of a policy and a `Trajectory`. We will use the following trajectory.

```julia
trajectory = Trajectory(
            CircularArraySARTSTraces(capacity = 1000, state = Float32 => (4,),action = Float32 => (1,)), 
            MetaSampler(
                actor = MultiBatchSampler(BatchSampler{(:state,)}(32), 10),
                critic = MultiBatchSampler(BatchSampler{SS′ART}(32), 1000)
            ),
            InsertSampleRatioController(ratio = 1/1000, threshold = 1000)
        ) 
```

MPO needs to store `SART` Traces, i.e. State-Action-Reward-Terminal-NextState. Here we use a fixed sized buffer with a capacity of 1000 steps. Then we specify the `Sampler`. MPO needs a specific type of sampler called a `MetaSampler`. A MetaSampler contains several named samplers, here one named `:actor` and the other `critic`. As you may have guessed, one samples to update the actor and the other for the critic (the QNetworks). You must use these exact names. Each Sampler must be a `MultiBatchSampler`, that will sample multiple batch to update the networks for several iterations. Here we update the critic 1000 times but only 10 times the policy. The actor sampler must sample only `(:state,)` traces, it does not need any other trace, the critic needs the `SS′ART` traces to perform the 1-step TD update on the `qnetwork`s. Here we sample batches of 32 transitions, of course this is a hyperparameter that you can tune to your liking.
Finally, we decide on the `InsertSampleRatioController`. We decide to start sampling to update the networks once we have inserted `threshold = 1000` transitions in the buffer (that is, when the buffer is full). You can chose another value but it does not make sense to pick one that is larger than the capacity of the buffer. Ratio defines how many steps are to be done between each sample call. In this case, we do 1000 steps to collect data before sampling and updating the networks. 

To summarize, with this setup, the algorithm will perform the following:
1. Interact 1000 times with the environment to fill the buffer.
2. Sample 1000 batches of 32 state-action-reward-terminal-next_state.
3. Update each qnetworks 500 times, once with each batch. 
4. Sample 10 batches of 32 states.
5. Update the actor 10 times.
6. Perform 1000 new steps with the new policy and replace the old ones in the buffer.
7. Unless the stopping criterion is true, go back to 2.

We can now create the agent, and run the experiment for 50,000 steps:
```julia
agent = Agent(policy = policy, trajectory = trajectory)
stop_condition = StopAfterStep(50_000, is_show_progress=true)
hook = TotalRewardPerEpisode()
run(agent, env, stop_condition, hook)
```

This should take a couple of minutes on a recent CPU. You can plot the result, for example with UnicodePlots:
```julia
using UnicodePlots
lineplot(hook.episodes, hook.mean_rewards, xlabel="episode", ylabel="mean episode reward", title = "Cartpole Continuous Action Space")
```

### Learning on a GPU

If you have a CUDA compatible GPU, you can accelerate your experiments by transfering the neural networks on the card. `MPOPolicy` comes with a method for the `gpu` function from the `Flux` package.

```julia
using CUDA

policy = gpu(policy) #Recreate a new policy if you already trained it.
agent = Agent(policy = policy, trajectory = trajectory)
stop_condition = StopAfterStep(50_000, is_show_progress=true)
hook = TotalRewardPerEpisode()
run(agent, env, stop_condition, hook) #Using the GPU is slower in this case because the NN and the batch size are small.
```

## Learning a discrete Cartpole policy

To use MPO with a discrete action space only requires simple changes. 
1. Instantiate the environment with `continuous = false`
2. Instead of using a `GaussianNetwork`, you should use the `CategoricalNetwork`. 
3. The action is now a one-hot vector of length two, because the action_size is 2.

## How to use the CovGaussianNetwork

`CovGaussianNetowrk` allows the approximation of a policy with a correlation between action dimensions, unlike the `GaussianNetwork` that only models a standard deviation for each dimension independently. In practice, this only requires two changes to the above example with `GaussianNetwork`:
1. Use a `CovGaussianNetowrk` instead of a `GaussianNetwork`.
2. The output size of the second head ($\Sigma$) should not be the action size ($|A|$), but $\frac{|A|*(|A|+1)}{2}$. For the Cartpole environment, the remains 1 since the action is of length 1.


