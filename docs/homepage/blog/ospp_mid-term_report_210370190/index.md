@def title = "Implement Multi-Agent Reinforcement Learning Algorithms in Julia"
@def description = """
    This is a technical report of the summer OSPP project [Implement Multi-Agent Reinforcement Learning Algorithms in Julia](https://summer.iscas.ac.cn/#/org/prodetail/210370190?lang=en). In this report, the following three parts are covered: the first section is a basic introduction to the project, the second section contains the implementation details of several multi-agent algorithms, and in the last section we discussed our future plan.
    """
@def is_enable_toc = true
@def has_code = true
@def has_math = true

@def front_matter = """
    {
        "authors": [
            {
                "author":"Peter Chen",
                "authorURL":"https://github.com/peterchen96",
                "affiliation":"ECNU",
                "affiliationURL":"http://english.ecnu.edu.cn/"
            }
        ],
        "publishedDate":"2021-08-13",
        "citationText":"Peter Chen, 2021"
    }"""

@def bibliography = "bibliography.bib"

## 1. Project Information

Recent advances in reinforcement learning led to many breakthroughs in artificial intelligence. Some of the latest deep reinforcement learning algorithms have been implemented in ReinforcementLearning.jl with Flux. Currently, we only have some CFR related algorithms implemented. We'd like to have more implemented, including MADDPG, COMA, NFSP, PSRO.

### Schedule

| Date       | Mission Content |
| :-----------: | :---------: |
| 07/01 -- 07/14 | Refer to the paper and the existing implementation to get familiar with the `NFSP` algorithm. |
| 07/15 -- 07/29 | Add `NFSP` algorithm into `RLZoo.jl`, and test it on the `KuhnPokerEnv`. |
| 07/30 -- 08/07 | Fix the existing bugs of `NFSP` and implement the `MADDPG` algorithm into `RLZoo.jl`. |
| 08/08 -- 08/15 | Update the `MADDPG` algorithm and test it on the `KuhnPokerEnv`,  also complete the **mid-term report**. |
| 08/16 -- 08/30 | Test `MADDPG` algorithm on more envs and consider implementing the `ED` algorithm into `RLZoo.jl`. |
| 08/31 -- 09/07 | Complete the `ED` implementation, and add relative experiments. |
| 09/08 -- 09/14 | Consider implementing `PSRO` algorithm into `RLZoo.jl`. |
| 09/15 -- 09/30 | Complete `PSRO` implementation and add relative experiments, also complete the **final-term report**. |

### Accomplished Work

From July 1st to now, I mainly have implemented the `Neural Fictitious Self-play`(NFSP) algorithm and added it into `ReinforcementLearningZoo.jl`(RLZoo.jl). A workable experiment is also added to the documentation. Besides, the `Multi-agent Deep Deterministic Policy Gradient`(MADDPG) algorithm's semi-finished implementation has been placed into `RLZoo.jl` and will test it on more envs in the next weeks. Related commits are listed below:

- [add Base.:(==) and Base.hash for AbstractEnv and test nash_conv on KuhnPokerEnv#348](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/348)
- [Supplement functions in ReservoirTrajectory and BehaviorCloningPolicy #390](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/390)
- [Implementation of NFSP and NFSP_KuhnPoker experiment #402](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/402)
- [correct nfsp implementation #439](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/439)
- [add MADDPG algorithm #444](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/444)

## 2. Implementation and Usage

This section will first briefly review the `Agent` structure defined in `ReinforcementLearning.jl`. Then I'll explain how I implemented `NFSP` and `MADDPG`, followed by a short example to demonstrate how others can use them in their customized environments.

### 2.1 An Introduction to `Agent`

The `Agent` struct is an extended `AbstractPolicy` that includes a concrete policy and a trajectory. The trajectory is used to collect the necessary information to train the policy. In the existing [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/policies/agents/agent.jl),  the lifecycle of the interactions between agents and environments is split into several stages, including `PreEpisodeStage`,  `PreActStage`, `PostActStage` and `PostEpisodeStage`.

```Julia
function (agent::Agent)(stage::AbstractStage, env::AbstractEnv)
    update!(agent.trajectory, agent.policy, env, stage)
    update!(agent.policy, agent.trajectory, env, stage)
end

function (agent::Agent)(stage::PreActStage, env::AbstractEnv, action)
    update!(agent.trajectory, agent.policy, env, stage, action)
    update!(agent.policy, agent.trajectory, env, stage)
end
```

And when running the experiment, based on the built-in [`run`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/639717388fb41199c98b90406bea76232bc6294d/src/ReinforcementLearningCore/src/core/run.jl#L16) function, the `agent` can update its policy and trajectory based on the behaviors that we have defined. Thanks to the `multiple dispatch` in Julia,  the **main focus** when implementing a new algorithm is how to **customize the behavior** of collecting the training information and updating the policy when at the specific stage. For more details, you can refer to this [blog](https://juliareinforcementlearning.org/blog/an_introduction_to_reinforcement_learning_jl_design_implementations_thoughts/#21_the_general_workflow).

### 2.2 Neural Fictitious Self-play(NFSP) algorithm

#### Brief Introduction

Neural Fictitious Self-play(NFSP)\dcite{DBLP:journals/corr/HeinrichS16} algorithm is a useful multi-agent algorithm that works well for imperfect-information games. Each agent who applies the `NFSP` algorithm will include one `Reinforcement Learning`(RL) agent and one `Supervised Learning`(SL) agent. **RL agent** works to find the best response to the state from the self-play process, and **SL agent** works to learn the best response from RL agent's policy. What's more, `NFSP` also uses two technical innovations to ensure stability, including [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) for SL agent and anticipatory dynamics\dcite{1406126} when training.

#### Implementation

In RLZoo.jl, I implement the [`NFSPAgent`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/nfsp/nfsp.jl) which define the `NFSPAgent` struct and design its behaviors according to the `NFSP` algorithm\dcite{DBLP:journals/corr/HeinrichS16}, including collecting needed information and how to update the policy. And the [`NFSPAgentManager`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/nfsp/nfsp_manager.jl) is a special multi-agent manager that all agents apply `NFSP` algorithm. Besides, the [`abstract_nfsp`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/nfsp/abstract_nfsp.jl) customize the `run` function for `NFSPAgentManager`.

Since the core of the algorithm is how to customize the `NFSPAgent`, the following content in this section will only be around it. The structure of `NFSPAgent` is as the following:
```Julia
mutable struct NFSPAgent <: AbstractPolicy
    rl_agent::Agent
    sl_agent::Agent
    η # anticipatory parameter
    rng
    update_freq::Int # update frequency
    update_step::Int # count the step
    mode::Bool # `true` for best response mode(RL agent's policy), `false` for  average policy mode(SL agent's policy). Only used in training.
end
```
Based on 2.1, the core of the `NFSPAgent` is customized behaviors on the specific stage:

- PreEpisodeStage

Here, `NFSPAgent` should set train mode based on the anticipatory dynamics\dcite{1406126} and delete the terminated state and dummy action if having gone through one episode before. Note that here deleting the terminated state and dummy action is necessary for the algo(see the [note](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/4e5d258798088b1c628401b6b9de18aa8cbb3ab3/src/ReinforcementLearningCore/src/policies/agents/agent.jl#L134)), otherwise may occur to have some unreliable samples.
```Julia
function (π::NFSPAgent)(stage::PreEpisodeStage, env::AbstractEnv, ::Any)
    # delete the terminal state and dummy action.
    update!(π.rl_agent.trajectory, π.rl_agent.policy, env, stage)

    # set the train's mode before the episode.(anticipatory dynamics)
    π.mode = rand(π.rng) < π.η
end
```

- PreActStage

In this stage, `NFSPAgent` should collect the personal information `state` and `action` to the RL agent's trajectory, and if on the `best response mode`, also update the SL agent's trajectory. Besides, if satisfying the condition of updating, here also need to update the inner agents. The code is just like the following:
```Julia
function (π::NFSPAgent)(stage::PreActStage, env::AbstractEnv, action)
    rl = π.rl_agent
    sl = π.sl_agent
    # update trajectory
    if π.mode
        update!(sl.trajectory, sl.policy, env, stage, action)
        rl(stage, env, action)
    else
        update!(rl.trajectory, rl.policy, env, stage, action)
    end

    # update policy
    π.update_step += 1
    if π.update_step % π.update_freq == 0
        if π.mode
            update!(sl.policy, sl.trajectory)
        else
            rl_learn!(rl.policy, rl.trajectory)
            update!(sl.policy, sl.trajectory)
        end
    end
end
```

- PostActStage

Here, the agent needs to collect the personal `reward` and  the `is_terminated` judgment of the current state to the RL agent's trajectory.
```Julia
function (π::NFSPAgent)(::PostActStage, env::AbstractEnv, player::Any)
    push!(π.rl_agent.trajectory[:reward], reward(env, player))
    push!(π.rl_agent.trajectory[:terminal], is_terminated(env))
end
```

- PostEpisodeStage

When one episode is terminated, the agent should collect the terminated state and a dummy action(see the [note](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/4e5d258798088b1c628401b6b9de18aa8cbb3ab3/src/ReinforcementLearningCore/src/policies/agents/agent.jl#L134)) to the RL agent's trajectory. Also, the reward and terminal judgment need to be corrected to avoid getting wrong samples when playing the sequential or terminal_reward games.
```Julia
function (π::NFSPAgent)(::PostEpisodeStage, env::AbstractEnv, player::Any)
    rl = π.rl_agent
    sl = π.sl_agent
    # update trajectory
    if !rl.trajectory[:terminal][end]
        rl.trajectory[:reward][end] = reward(env, player)
        rl.trajectory[:terminal][end] = is_terminated(env)
    end

    action = rand(action_space(env, player))
    push!(rl.trajectory[:state], state(env, player))
    push!(rl.trajectory[:action], action)
    if haskey(rl.trajectory, :legal_actions_mask)
        push!(rl.trajectory[:legal_actions_mask], legal_action_space_mask(env, player))
    end
    
    # update the policy    
    ...
end
```

#### Usage

According to the paper\dcite{DBLP:journals/corr/HeinrichS16}, here, the RL agent is default as `QBasedPolicy` with `CircularArraySARTTrajectory.` The SL agent is default as `BehaviorCloningPolicy` with `ReservoirTrajectory.` So you can customize the agent under the restriction and test the algo on any interested multi-agent game. **Note that** if the game's states can't be used as the network's input, you need to [wrap](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/50756bbe9e1925a9320d1abdbbc6255c1b4a27f1/src/ReinforcementLearningEnvironments/src/environments/wrappers/StateTransformedEnv.jl#L9) them before using the algorithm.

Here is one experiment [`JuliaRL_NFSP_KuhnPoker.jl`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl) as one usage example, which tests the algorithm on the Kuhn Poker game. Since the type of states in the existing [`KuhnPokerEnv`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/639717388fb41199c98b90406bea76232bc6294d/src/ReinforcementLearningEnvironments/src/environments/examples/KuhnPokerEnv.jl#L1) in `ReinforcementLearningEnvironments.jl` is the `tuple` of symbols, I simply encode the state just like the following:
```Julia
env = KuhnPokerEnv()
wrapped_env = StateTransformedEnv(
    env;
    state_mapping = s -> [findfirst(==(s), state_space(env))],
    state_space_mapping = ss -> [[findfirst(==(s), state_space(env))] for s in state_space(env)]
    )
```

In this experiment, `RL agent` use [`DQNLearner`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/50756bbe9e1925a9320d1abdbbc6255c1b4a27f1/src/ReinforcementLearningZoo/src/algorithms/dqns/dqn.jl#L23) to learn the best response:
```Julia
rl_agent = Agent(
    policy = QBasedPolicy(
        learner = DQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 64, relu; init = glorot_normal(rng)),
                    Dense(64, na; init = glorot_normal(rng))
                ) |> cpu,
                optimizer = Descent(0.01),
            ),
            target_approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 64, relu; init = glorot_normal(rng)),
                    Dense(64, na; init = glorot_normal(rng))
                ) |> cpu,
            ),
            γ = 1.0f0,
            loss_func = huber_loss,
            batch_size = 128,
            update_freq = 128,
            min_replay_history = 1000,
            target_update_freq = 1000,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :linear,
            ϵ_init = 0.06,
            ϵ_stable = 0.001,
            decay_steps = 1_000_000,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 200_000,
        state = Vector{Int} => (ns, ),
    ),
)
```

And the `SL agent` is defined as the following:
```Julia
sl_agent = Agent(
    policy = BehaviorCloningPolicy(;
        approximator = NeuralNetworkApproximator(
            model = Chain(
                    Dense(ns, 64, relu; init = glorot_normal(rng)),
                    Dense(64, na; init = glorot_normal(rng))
                ) |> cpu,
            optimizer = Descent(0.01),
        ),
        explorer = WeightedSoftmaxExplorer(),
        batch_size = 128,
        min_reservoir_history = 1000,
        rng = rng,
    ),
    trajectory = ReservoirTrajectory(
        2_000_000;# reservoir capacity
        rng = rng,
        :state => Vector{Int},
        :action => Int,
    ),
)
```

Based on the defined agents, the `NFSPAgentManager` can be customized as the following:
```Julia
nfsp = NFSPAgentManager(
    Dict(
        (player, NFSPAgent(
            deepcopy(rl_agent),
            deepcopy(sl_agent),
            0.1f0, # anticipatory parameter
            rng,
            128, # update_freq
            0, # initial update_step
            true, # initial NFSPAgent's learn mode
        )) for player in players(wrapped_env) if player != chance_player(wrapped_env)
    )
)
```

Based on the setting [`stop_condition`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/50756bbe9e1925a9320d1abdbbc6255c1b4a27f1/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl#L126) and designed [`hook`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/50756bbe9e1925a9320d1abdbbc6255c1b4a27f1/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl#L15), you can just `run(nfsp, wrapped_env, stop_condition, hook)` to perform the experiment. The **result** of the experiment is just like the following.

\dfig{body;JuliaRL_NFSP_KuhnPoker.png;Result of the experiment.}

### 2.3 Multi-agent Deep Deterministic Policy Gradient(MADDPG) algorithm

#### Brief Introduction

The Multi-agent Deep Deterministic Policy Gradient(MADDPG)\dcite{DBLP:journals/corr/LoweWTHAM17} algorithm improves the [Deep Deterministic Policy Gradient(DDPG)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), which works well on multi-agent games. Based on the DDPG, the critic of each agent in MADDPG can get all agents' policies according to the paper's hypothesis\dcite{DBLP:journals/corr/LoweWTHAM17}, including their personal states and actions, which can help get a more reasonable score of the actor's policy.

#### Implementation

Since there has been [`DDPGPolicy`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/50756bbe9e1925a9320d1abdbbc6255c1b4a27f1/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/ddpg.jl#L47) in the RLZoo, I implement the [`MADDPGManager`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/maddpg.jl) which is a special multi-agent manager that all agents apply `DDPGPolicy` with one **improved critic**. The structure of `MADDPGManager` is as the following:
```Julia
mutable struct MADDPGManager{P<:DDPGPolicy, T<:AbstractTrajectory, N<:Any} <: AbstractPolicy
    agents::Dict{<:N, <:Agent{<:NamedPolicy{<:P, <:N}, <:T}}
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::AbstractRNG
end
```

Each agent in the MADDPGManager uses `DDPGPolicy` with one trajectory, which collects their own information. Here [`NamedPolicy`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/policies/agents/named_policy.jl) is a useful substruct of `AbstractPolicy` when meeting the multi-agent games, which combine the player's name and detailed policy. So that can use `Agent` 's [default behaviors for known trajectories](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/b0b8e8236524a7af0a2da8987ae2261c257f94b2/src/ReinforcementLearningCore/src/policies/agents/agent.jl#L85) to collect the necessary information. 

As for updating the policy, the process is mainly the same as the [`DDPGPolicy`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/b0b8e8236524a7af0a2da8987ae2261c257f94b2/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/ddpg.jl#L139), apart from each agent's critic will assemble all agents' personal states and actions. For more details, can refer to the [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/maddpg.jl).

#### Usage

Here `MADDPG` is used for simultaneous games, or you can drop the dummy action of other players when [wrapping](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/50756bbe9e1925a9320d1abdbbc6255c1b4a27f1/src/ReinforcementLearningEnvironments/src/environments/wrappers/ActionTransformedEnv.jl#L9) the sequential game. And there is one experiment [`JuliaRL_MADDPG_KuhnPoker.jl`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl) as one usage example, which tests the algorithm on the Kuhn Poker game. Since the Kuhn Poker is one sequential game, I wrap the game just like the following:
```Julia
wrapped_env = ActionTransformedEnv(
        StateTransformedEnv(
            env;
            state_mapping = s -> [findfirst(==(s), state_space(env))],
            state_space_mapping = ss -> [[findfirst(==(s), state_space(env))] for s in state_space(env)]
            ),
        ## drop the dummy action of the other agent.
        action_mapping = x -> length(x) == 1 ? x : Int(x[current_player(env)] + 1),
    )
```

And customize the actor and critic's network:
```Julia
ns, na = 1, 1
n_players = 2

create_actor() = Chain(
        Dense(ns, 64, relu; init = glorot_uniform(rng)),
        Dense(64, 64, relu; init = glorot_uniform(rng)),
        Dense(64, na, tanh; init = glorot_uniform(rng)),
    )

create_critic() = Chain(
    Dense(n_players * ns + n_players * na, 64, relu; init = glorot_uniform(rng)),
    Dense(64, 64, relu; init = glorot_uniform(rng)),
    Dense(64, 1; init = glorot_uniform(rng)),
    )
```

So that can design the fundamental policy and trajectory like the following:
```Julia
policy = DDPGPolicy(
    behavior_actor = NeuralNetworkApproximator(
        model = create_actor(),
        optimizer = ADAM(),
    ),
    behavior_critic = NeuralNetworkApproximator(
        model = create_critic(),
        optimizer = ADAM(),
    ),
    target_actor = NeuralNetworkApproximator(
        model = create_actor(),
        optimizer = ADAM(),
    ),
    target_critic = NeuralNetworkApproximator(
        model = create_critic(),
        optimizer = ADAM(),
    ),
    γ = 0.99f0,
    ρ = 0.995f0,
    na = na,
    start_steps = 1000,
    start_policy = RandomPolicy(-0.9..0.9; rng = rng),
    update_after = 1000,
    act_limit = 0.9,
    act_noise = 0.1,
    rng = rng,
)
trajectory = CircularArraySARTTrajectory(
    capacity = 10000, # replay buffer capacity
    state = Vector{Int} => (ns, ),
    action = Float32 => (na, ),
)
```

Based on the above policy and trajectory, the `MADDPGManager` can be defined as the following:
```Julia
agents = MADDPGManager(
    Dict((player, Agent(
        policy = NamedPolicy(player, deepcopy(policy)),
        trajectory = deepcopy(trajectory),
    )) for player in players(env) if player != chance_player(env)),
    128, # batch_size
    128, # update_freq
    0, # update_step
    rng
)
```

Plus on the [`stop_condition`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/50756bbe9e1925a9320d1abdbbc6255c1b4a27f1/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl#L110) and [`hook`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/50756bbe9e1925a9320d1abdbbc6255c1b4a27f1/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl#L15), you can experiment by `run(agents, wrapped_env, stop_condition, hook)`. The **result** of the experiment is just like the following:

\dfig{body;JuliaRL_MADDPG_KuhnPoker.png;Result of the experiment.}

**Note that** the current `MADDPG` still can only work on the envs of `MINIMAL_ACTION_SET,` i.e., all actions in the environment's action space are legal. And the Kuhn Poker game may not be suitable for the test since `MADDPG` is one deterministic algorithm that the state's response is one deterministic action. In the next weeks, I'll update the algorithm and try to test it on other games.

## 3. Reviews and Future Plan

### 3.1 Reviews

From applying the project to now, since spending much time on getting familiar with the algorithm and structure of RL.jl, my progress was slow in the initial weeks. However, thanks to the mentor's patience in leading, I realize the convenience of the general workflow in RL.jl and improve my comprehension of the algorithm.

### 3.2 Future Plan

In the first section's `Schedule`, I have listed a draft plan for the next serval weeks. In detail, I want to complete the following missions:

- Test `MADDPG` on more suitable envs and add relative experiments. (08/16 - 08/23)
- Consider implementing the `Exploitability Descent`(ED) algorithm and add related experiments. (08/24 - 09/07)
- Consider implementing the `Policy-Spaced Response Oracles`(PSRO) algorithm and add related experiments. (09/08 - 09/22)
- Fix the existing bugs of algorithms and finish the **final-term report**. (09/23 - 09/30)
