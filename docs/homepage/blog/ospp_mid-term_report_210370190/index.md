@def title = "Implement Multi-Agent Reinforcement Learning Algorithms in Julia (Summer OSPP Project 210370190) Mid-term Report"
@def description = """
    This is a mid-term report of the summer ospp project [Implement Multi-Agent Reinforcement Learning Algorithms in Julia](https://summer.iscas.ac.cn/#/org/prodetail/210370190?lang=en). The report includes the following three components: `Project Information`, `Implementation and Usage of Algorithms` and `Reviews and Future Plan`.
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
                "affiliation":"",
                "affiliationURL":""
            }
        ],
        "publishedDate":"2021-08-13",
        "citationText":"Peter Chen, 2021"
    }"""

@def bibliography = "bibliography.bib"

## 1. Project Information

### Project Name

Implement Multi-Agent Reinforcement Learning Algorithms in Julia

### Scheme Description

Recent advances in reinforcement learning led to many breakthroughs in artificial intelligence. Some of the latest deep reinforcement learning algorithms have been implemented in ReinforcementLearning.jl with Flux. Currently, we only have some CFR related algorithms implemented. We'd like to have more implemented, including MADDPG, COMA, NFSP, PSRO.

### Time Planning

| Date       | Mission Content |
| :-----------: | :---------: |
| 07/01 - 07/14 | Refer the paper and the existing implement to get familiar with the `NFSP` algorithm. |
| 07/15 - 07/29 | Add `NFSP` algorithm into `RLZoo.jl`, and test it on the `KuhnPokerEnv`. |
| 07/30 - 08/07 | Fix the existing bugs of `NFSP` and implement `MADDPG` algorithm into `RLZoo.jl`. |
| 08/08 - 08/15 | Update the `MADDPG` algo and test it on the `KuhnPokerEnv`,  also complete the **mid-term report**. |
| 08/16 - 08/30 | Test `MADDPG` algo on more envs and consider implementing `ED` algorithm into `RLZoo.jl`. |
| 08/31 - 09/07 | Complete the `ED` implementation, and add relative experiments. |
| 09/08 - 09/14 | Consider implementing `PSRO` algorithm into `RLZoo.jl`. |
| 09/15 - 09/30 | Complete `PSRO` implementation and add relative experiments, also complete the **final-term report**. |

### Accomplished Work

From July 1st to now, I mainly have implemented the `Neural Fictitious Self-play`(NFSP) algorithm into `ReinforcementLearningZoo.jl`(RLZoo.jl) and add one relative experiment in the documentation. Also `Multi-agent Deep Deterministic Policy Gradient`(MADDPG) algorithm's semi-finished implementation has been placed into `RLZoo.jl` and will test it on more envs in the next weeks. Related commits list as the following:

- [add Base.:(==) and Base.hash for AbstractEnv and test nash_conv on KuhnPokerEnv#348](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/348)
- [Supplement functions in ReservoirTrajectory and BehaviorCloningPolicy #390](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/390)
- [Implementation of NFSP and NFSP_KuhnPoker experiment #402](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/402)
- [correct nfsp implementation #439](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/439)
- [add MADDPG algorithm #444](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/444)

## 2. Implementation and usage of Algorithms

This section will first briefly introduce the `Agent` struct and its usage, and then introduce the details about the implementation and usage of `NFSP` and `MADDPG.`

### 2.1 Introduction of `Agent`

`Agent` struct is an extended `AbstractPolicy` that includes the detailed policy and one trajectory, which collect the necessary information for training the policy. In the existing [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/policies/agents/agent.jl), there have been defined default behaviors(like the following) when self-playing the game, which splits the updating process of the strategy into several stages, including `PreEpisodeStage`,  `PreActStage`, `PostActStage` and `PostEpisodeStage`. 

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

And when running the experiment, based on the built-in [`run`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/639717388fb41199c98b90406bea76232bc6294d/src/ReinforcementLearningCore/src/core/run.jl#L16) function(or can define a new `run` function for the algo if necessary), the `agent` can update its policy and trajectory based on the behaviors that we have defined. Thanks to the `multiple dispatch` in Julia,  the **main focus** when implementing the algo is that consider how to **customize the behavior** about collecting the training information and updating the policy when on the specific stage. More details can be referred to the [blog](https://juliareinforcementlearning.org/blog/an_introduction_to_reinforcement_learning_jl_design_implementations_thoughts/#21_the_general_workflow).

### 2.2 Neural Fictitious Self-play(NFSP) algorithm

#### Brief Introduction
Neural Fictitious Self-play(NFSP)\dcite{DBLP:journals/corr/HeinrichS16} algorithm is a useful multi-agent algorithm that works well for imperfect-information games. Each agent who applies the `NFSP` algo will include one `Reinforcement Learning`(RL) agent and one `Supervised Learning`(SL) agent. **RL agent** works to find the best response to the state from the self-play process, and **SL agent** works to learn the best response from RL agent's policy. What's more, `NFSP` also uses two technical innovations to ensure stability, including [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling) for SL agent and anticipatory dynamics when training.

#### Implementation
In RLZoo.jl, I implement the [`NFSPAgent`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/nfsp/nfsp.jl) which define the `NFSPAgent` struct and design the behaviors about it according to the `NFSP` algo\dcite{DBLP:journals/corr/HeinrichS16}, including collecting needed information and how to update the policy. And the [`NFSPAgentManager`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/nfsp/nfsp_manager.jl) is a special multi-agent manager that all agents apply `NFSP` algo. Besides, the [`abstract_nfsp`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/nfsp/abstract_nfsp.jl) customize the `run` function for `NFSPAgentManager`.

Since the core of the algo is how to customize the `NFSPAgent`, the following content in this section will only be around it. The structure of `NFSPAgent` is as the following:
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

Here, `NFSPAgent` should set train mode based on the anticipatory dynamics and delete the terminated state and dummy action if having gone through one episode before. Note that here deleting the terminated state and dummy action is necessary for the algo(see the [note](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/4e5d258798088b1c628401b6b9de18aa8cbb3ab3/src/ReinforcementLearningCore/src/policies/agents/agent.jl#L134)), otherwise may occur to have some unreliable samples.
```Julia
function (π::NFSPAgent)(stage::PreEpisodeStage, env::AbstractEnv, ::Any)
    # delete the terminal state and dummy action.
    update!(π.rl_agent.trajectory, π.rl_agent.policy, env, stage)

    # set the train's mode before the episode.(anticipatory dynamics)
    π.mode = rand(π.rng) < π.η
end
```

- PreActStage

In this stage, `NFSPAgent` should collect the personal information `state` and `action` to the RL agent's trajectory, and if on the `best response mode`, also update the SL agent's trajectory. Besides, if satisfying the condition of updating, here also need to update the inner agents. The rough code is just like the following:
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

According to the paper\dcite{DBLP:journals/corr/HeinrichS16}, here, the RL agent is default as `QBasedPolicy` with `CircularArraySARTTrajectory,` and the SL agent is default as `BehaviorCloningPolicy` with `ReservoirTrajectory.` So you can customize the agent under the restriction, and test the algo on any interested multi-agent game.

Here is one experiment [`JuliaRL_NFSP_KuhnPoker.jl`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl) as one usage example, which tests the algo on the Kuhn Poker game. The result of the experiment is just like the following.

\dfig{body;JuliaRL_NFSP_KuhnPoker.png;Result of the experiment.}

### 2.3 Multi-agent Deep Deterministic Policy Gradient(MADDPG) algorithm

#### Brief Introduction
The Multi-agent Deep Deterministic Policy Gradient(MADDPG)\dcite{DBLP:journals/corr/LoweWTHAM17} algorithm improves the Deep Deterministic Policy Gradient(DDPG), which works well on multi-agent games. Based on the DDPG, the critic of each agent in MADDPG can get all agents' policies according to the paper's hypothesis\dcite{DBLP:journals/corr/LoweWTHAM17}, including their personal states and actions, which can help get a more reasonable score of the actor's policy.

#### Implementation
Since there has been `DDPGPolicy` in the RLZoo, I implement the [`MADDPGManager`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/maddpg.jl) which is a special multi-agent manager that all agents apply `DDPGPolicy` with one **improved critic**. The structure of `MADDPGManager` is as the following:
```Julia
mutable struct MADDPGManager{P<:DDPGPolicy, T<:AbstractTrajectory, N<:Any} <: AbstractPolicy
    agents::Dict{<:N, <:Agent{<:NamedPolicy{<:P, <:N}, <:T}}
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::AbstractRNG
end
```
Where each agent in the MADDPGManager uses `DDPGPolicy` with one trajectory, which collects their own information. Here [`NamedPolicy`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/policies/agents/named_policy.jl) is a useful substruct of `AbstractPolicy` when meeting the multi-agent games, which combine the player's name and detailed policy. So that can use `Agent` 's [default behaviors for known trajectories](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/b0b8e8236524a7af0a2da8987ae2261c257f94b2/src/ReinforcementLearningCore/src/policies/agents/agent.jl#L85) to collect the necessary information. 

As for updating the policy, the process is mainly the same as the [`DDPGPolicy`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/b0b8e8236524a7af0a2da8987ae2261c257f94b2/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/ddpg.jl#L139), apart from each agent's critic will assemble all agents' personal states and actions. For more details, can refer to the [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/maddpg.jl).

#### Usage
Here is one experiment [`JuliaRL_MADDPG_KuhnPoker.jl`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl) as one usage example, which tests the algo on the Kuhn Poker game. The result of the experiment is just like the following.

\dfig{body;JuliaRL_MADDPG_KuhnPoker.png;Result of the experiment.}

**Note that** the current `MADDPG` still can only work on the envs of `MINIMAL_ACTION_SET,` i.e., all actions in the environment's action space are legal. And the Kuhn Poker game may not be suitable for the test of the algo. In the next weeks, I'll update the algo and try to test it on other games.

## 3. Reviews and Future Plan

### 3.1 Reviews

From applying the project to now, since spending much time on getting familiar with the algorithm and structure of RL.jl, my progress was slow in the initial weeks. However, thanks to the mentor's patience in leading, I realize the convenience of the general workflow in RL.jl and improve my comprehension of the algo.

### 3.2 Future Plan

In the `Time Planning`, I have listed a draft plan for the next serval weeks. In detail, I want to complete the following missions:

- Test `MADDPG` on more suitable envs and add relative experiments. (08/16 - 08/23)
- Consider implementing the `Exploitability Descent`(ED) algorithm and add related experiments. (08/24 - 09/07)
- Consider implementing the `Policy-Spaced Response Oracles`(PSRO) algorithm and add related experiments. (09/08 - 09/22)
- Fix the existing bugs of algorithms and finish the **final-term report**. (09/23 - 09/30)

