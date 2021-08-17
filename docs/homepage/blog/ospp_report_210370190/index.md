@def title = "Implement Multi-Agent Reinforcement Learning Algorithms in Julia"
@def description = """
    This is a technical report of the summer OSPP project [Implement Multi-Agent Reinforcement Learning Algorithms in Julia](https://summer.iscas.ac.cn/#/org/prodetail/210370190?lang=en). In this report, the following two parts are covered: the first section is a basic introduction to the project, and the second section contains the implementation details of several multi-agent algorithms.
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
        "publishedDate":"2021-08-17",
        "citationText":"Peter Chen, 2021"
    }"""

@def bibliography = "bibliography.bib"

## 1. Project Information

Recent advances in reinforcement learning led to many breakthroughs in artificial intelligence. Some of the latest deep reinforcement learning algorithms have been implemented in [ReinforcementLearning.jl](https://juliareinforcementlearning.org/) with [Flux](https://fluxml.ai/). Currently, we only have some [CFR related algorithms](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningZoo/src/algorithms/cfr) implemented. We'd like to have more implemented, including **MADDPG**\dcite{DBLP:journals/corr/LoweWTHAM17}, **COMA**\dcite{DBLP:journals/corr/FoersterFANW17}, **NFSP**\dcite{DBLP:journals/corr/HeinrichS16}, **PSRO**\dcite{DBLP:journals/corr/abs-1909-12823}.

### Schedule

| Date | Mission Content |
| :-----------: | :---------: |
| 07/01 -- 07/14 | Refer to the paper\dcite{DBLP:journals/corr/HeinrichS16} and the existing implementation to get familiar with the **NFSP** algorithm. |
| 07/15 -- 07/29 | Add **NFSP** algorithm into [ReinforcementLearningZoo.jl](https://juliareinforcementlearning.org/docs/rlzoo/), and test it on the [`KuhnPokerEnv`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.KuhnPokerEnv). |
| 07/30 -- 08/07 | Fix the existing bugs of **NFSP** and implement the **MADDPG** algorithm into ReinforcementLearningZoo.jl. |
| 08/08 -- 08/15 | Update the **MADDPG** algorithm and test it on the `KuhnPokerEnv`, also complete the **mid-term report**. |
| 08/16 -- 08/23 | Add support for environments of [`FULL_ACTION_SET`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.FULL_ACTION_SET) in **MADDPG** and test it on more games, such as [`simple_adversary`](https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/scenarios/simple_adversary.py). |
| 08/24 -- 08/30 | ... |

### Accomplished Work

From July 1st to now, I have implemented the **Neural Fictitious Self-play(NFSP)** algorithm and added it into [ReinforcementLearningZoo.jl](https://juliareinforcementlearning.org/docs/rlzoo/). A workable [experiment](https://juliareinforcementlearning.org/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker/#JuliaRL\\_NFSP\\_KuhnPoker) is also added to the documentation. Besides, the **Multi-agent Deep Deterministic Policy Gradient(MADDPG)** algorithm's semi-finished implementation has been placed into ReinforcementLearningZoo.jl, and I will test it on more envs in the next weeks. Related commits are listed below:

- [add Base.:(==) and Base.hash for AbstractEnv and test nash_conv on KuhnPokerEnv#348](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/348)
- [Supplement functions in ReservoirTrajectory and BehaviorCloningPolicy #390](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/390)
- [Implementation of NFSP and NFSP_KuhnPoker experiment #402](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/402)
- [correct nfsp implementation #439](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/439)
- [add MADDPG algorithm #444](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/444)
- ...

## 2. Implementation and Usage

In this section, I will first briefly review the [`Agent`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent) structure defined in [ReinforcementLearningCore.jl](https://juliareinforcementlearning.org/docs/rlcore/). Then I'll explain how these multi-agent algorithms(**NFSP**, **MADDPG**, ...) are implemented, followed by a short example to demonstrate how others can use them in their customized environments.

### 2.1 An Introduction to `Agent`

The [`Agent`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent) struct is an extended [`AbstractPolicy`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.AbstractPolicy) that includes a concrete policy and a trajectory. The trajectory is used to collect the necessary information to train the policy. In the existing [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/policies/agents/agent.jl),  the lifecycle of the [interactions](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent-Tuple{AbstractStage,%20AbstractEnv}) between agents and environments is split into several stages, including `PreEpisodeStage`,  `PreActStage`, `PostActStage` and `PostEpisodeStage`.

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

And when running the experiment, based on the built-in [`run`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningCore._run) function, the agent can update its policy and trajectory based on the behaviors that we have defined. Thanks to the [**multiple dispatch**](https://en.wikipedia.org/wiki/Multiple_dispatch) in Julia,  the **main focus** when implementing a new algorithm is how to **customize the behavior** of collecting the training information and updating the policy when in the specific stage. For more details, you can refer to this [blog](https://juliareinforcementlearning.org/blog/an_introduction_to_reinforcement_learning_jl_design_implementations_thoughts/#21_the_general_workflow).

### 2.2 Neural Fictitious Self-play(NFSP) algorithm

#### Brief Introduction

**Neural Fictitious Self-play(NFSP)**\dcite{DBLP:journals/corr/HeinrichS16} algorithm is a useful multi-agent algorithm that works well on imperfect-information games. Each agent who applies the **NFSP** algorithm has two inner agents, a **Reinforcement Learning (RL)** agent and a **Supervised Learning (SL)** agent. The **RL** agent is to find the best response to the state from the self-play process, and the **SL** agent is to learn the best response from the **RL** agent's policy. More importantly, **NFSP** also uses two technical innovations to ensure stability, including [**reservoir sampling**](https://en.wikipedia.org/wiki/Reservoir_sampling) for **SL** agent and **anticipatory dynamics**\dcite{1406126} when training. The following figure(from the paper\dcite{DBLP:journals/corr/abs-2104-10845}) shows the overall structure of **NFSP**(one agent).

\dfig{body;NFSP.png;The overall structure of **NFSP**(one agent).}

#### Implementation

In ReinforcementLearningZoo.jl, I implement the [`NFSPAgent`](https://juliareinforcementlearning.org/docs/rlzoo/#:~:text=ReinforcementLearningZoo.NFSPAgent) which define the `NFSPAgent` struct and design its behaviors according to the **NFSP** algorithm, including collecting needed information and how to update the policy. And the [`NFSPAgentManager`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.NFSPAgentManager) is a special multi-agent manager that all agents apply **NFSP** algorithm. Besides, in the [`abstract_nfsp`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/nfsp/abstract_nfsp.jl), I customize the `run` function for `NFSPAgentManager`.

Since the core of the algorithm is to define the behavior of the `NFSPAgent`, I'll explain how it is done as the following:
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

Based on our discussion in section 2.1, the core of the `NFSPAgent` is to customize its behavior in different stages:

- PreEpisodeStage

Here, the `NFSPAgent` should be set to the training mode based on the **anticipatory dynamics**. Besides, the **terminated state** and **dummy action** of the last episode must be removed at the beginning of each episode. (see the [note](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/4e5d258798088b1c628401b6b9de18aa8cbb3ab3/src/ReinforcementLearningCore/src/policies/agents/agent.jl#L134))
```Julia
function (π::NFSPAgent)(stage::PreEpisodeStage, env::AbstractEnv, ::Any)
    # delete the terminal state and dummy action.
    update!(π.rl_agent.trajectory, π.rl_agent.policy, env, stage)

    # set the train's mode before the episode.(anticipatory dynamics)
    π.mode = rand(π.rng) < π.η
end
```

- PreActStage

In this stage, the `NFSPAgent` should collect the personal information of **state** and **action**, and add them into the **RL** agent's trajectory. If it is set to the `best response mode`, we also need to update the **SL** agent's trajectory. Besides, if the condition of updating is satisfied, the inner agents also need to be updated. The code is just like the following:
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

After executing the action, the `NFSPAgent` needs to add the personal **reward** and the **is_terminated** results of the current state into the **RL** agent's trajectory.
```Julia
function (π::NFSPAgent)(::PostActStage, env::AbstractEnv, player::Any)
    push!(π.rl_agent.trajectory[:reward], reward(env, player))
    push!(π.rl_agent.trajectory[:terminal], is_terminated(env))
end
```

- PostEpisodeStage

When one episode is terminated, the agent should push the **terminated state** and a **dummy action** (see also the [note](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/4e5d258798088b1c628401b6b9de18aa8cbb3ab3/src/ReinforcementLearningCore/src/policies/agents/agent.jl#L134)) into the **RL** agent's trajectory. Also, the **reward** and **is_terminated** results need to be corrected to avoid getting the wrong samples when playing the games of [`SEQUENTIAL`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.SEQUENTIAL) or [`TERMINAL_REWARD`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.TERMINAL_REWARD).
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
    ...# here is the same as PreActStage `update the policy` part.
end
```

#### Usage

According to the paper\dcite{DBLP:journals/corr/HeinrichS16}, by default the **RL** agent is as [`QBasedPolicy`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.QBasedPolicy) with [`CircularArraySARTTrajectory`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.CircularArraySARTTrajectory-Tuple{}). And the **SL** agent is default as [`BehaviorCloningPolicy`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.BehaviorCloningPolicy-Union{Tuple{},%20Tuple{A}}%20where%20A) with [`ReservoirTrajectory`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/policies/agents/trajectories/reservoir_trajectory.jl). So you can customize the agent under the restriction and test the algorithm on any interested multi-agent games. **Note that** if the game's states can't be used as the network's input, you need to add a [state-related wrapper](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.StateTransformedEnv-Tuple{Any}) to the environment before applying the algorithm.

Here is one [experiment](https://juliareinforcementlearning.org/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker/#JuliaRL\\_NFSP\\_KuhnPoker) `JuliaRL_NFSP_KuhnPoker` as one usage example, which tests the algorithm on the Kuhn Poker game. Since the type of states in the existing [`KuhnPokerEnv`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.KuhnPokerEnv) is the `tuple` of symbols, I simply encode the state just like the following:
```Julia
env = KuhnPokerEnv()
wrapped_env = StateTransformedEnv(
    env;
    state_mapping = s -> [findfirst(==(s), state_space(env))],
    state_space_mapping = ss -> [[findfirst(==(s), state_space(env))] for s in state_space(env)]
    )
```

In this experiment, **RL** agent use [`DQNLearner`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.DQNLearner-Union{Tuple{},%20Tuple{Tf},%20Tuple{Tt},%20Tuple{Tq}}%20where%20{Tq,%20Tt,%20Tf}) to learn the best response:
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

And the **SL** agent is defined as the following:
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

Based on the defined inner agents, the `NFSPAgentManager` can be customized as the following:
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
            true, # initial NFSPAgent's training mode
        )) for player in players(wrapped_env) if player != chance_player(wrapped_env)
    )
)
```

Based on the setting [`stop_condition`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl#L126) and designed [`hook`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl#L15) in the experiment, you can just `run(nfsp, wrapped_env, stop_condition, hook)` to perform the experiment. Use [`Plots.plot`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl#L136) to get the following result: (here [`nash_conv`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/cfr/nash_conv.jl#L1) is one common metric to show the performance of a multi-agent reinforcement learning algorithm.)

\dfig{body;JuliaRL_NFSP_KuhnPoker.png;Result of the experiment.}

### 2.3 Multi-agent Deep Deterministic Policy Gradient(MADDPG) algorithm

#### Brief Introduction

The **Multi-agent Deep Deterministic Policy Gradient(MADDPG)**\dcite{DBLP:journals/corr/LoweWTHAM17} algorithm improves the [Deep Deterministic Policy Gradient(DDPG)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), which also works well on multi-agent games. Based on the DDPG, the critic of each agent in **MADDPG** can get all agents' policies according to the paper\dcite{DBLP:journals/corr/LoweWTHAM17}'s hypothesis, including their personal states and actions, which can help to get a more reasonable score of the actor's policy. The following figure(from the paper\dcite{8846699}) illustrates the framework of **MADDPG**.

\dfig{body;MADDPG.png;The framework of **MADDPG**.}

#### Implementation

Given that the [`DDPGPolicy`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.DDPGPolicy-Tuple{}) is already implemented in the ReinforcementLearningZoo.jl, I implement the [`MADDPGManager`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.MADDPGManager) which is a special multi-agent manager that all agents apply `DDPGPolicy` with one **improved critic**. The structure of `MADDPGManager` is as the following:
```Julia
mutable struct MADDPGManager{P<:DDPGPolicy, T<:AbstractTrajectory, N<:Any} <: AbstractPolicy
    agents::Dict{<:N, <:Agent{<:NamedPolicy{<:P, <:N}, <:T}}
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::AbstractRNG
end
```

Each agent in the `MADDPGManager` uses `DDPGPolicy` with one trajectory, which collects their own information. Here [`NamedPolicy`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.NamedPolicy) is a useful substruct of `AbstractPolicy` when meeting the multi-agent games, which combine the player's name and detailed policy. So that can use `Agent` 's [default behaviors](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent-Tuple{AbstractStage,%20AbstractEnv}) to collect the necessary information. 

As for updating the policy, the process is mainly the same as the [`DDPGPolicy`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/ddpg.jl#L139), apart from each agent's critic will assemble all agents' personal states and actions. For more details, you can refer to the [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/maddpg.jl#L59).

#### Usage

Here `MADDPGManager` is used for simultaneous games, or you can add an [action-related wrapper](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.ActionTransformedEnv-Tuple{Any}) to the sequential game to drop the dummy action of other players. And there is one [experiment](https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker/#JuliaRL\\_MADDPG\\_KuhnPoker) `JuliaRL_MADDPG_KuhnPoker` as one usage example, which tests the algorithm on the Kuhn Poker game. Since the Kuhn Poker is one sequential game, I wrap the game just like the following:
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

And customize the following actor and critic's network:
```Julia
rng = StableRNG(123)
ns, na = 1, 1 # dimension of the state and action.
n_players = 2 # the number of players

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

So that can design the inner `DDPGPolicy` and trajectory like the following:
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

Plus on the [`stop_condition`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl#L110) and [`hook`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl#L15) in the experiment, you can also `run(agents, wrapped_env, stop_condition, hook)` to perform the experiment. Use [`Plots.scatter`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl#L119) to get the following result:

\dfig{body;JuliaRL_MADDPG_KuhnPoker.png;Result of the experiment.}

**Note that** the current `MADDPGManager` still only works on the envs of [`MINIMAL_ACTION_SET`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.MINIMAL_ACTION_SET). And since **MADDPG** is one deterministic algorithm, i.e., the state's response is one deterministic action, the Kuhn Poker game may not be suitable for testing the performance. In the next weeks, I'll update the algorithm and try to test it on other games.
