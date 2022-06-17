@def title = "Implement Multi-Agent Reinforcement Learning Algorithms in Julia"
@def description = """
    This is a technical report of the summer OSPP project [Implement Multi-Agent Reinforcement Learning Algorithms in Julia](https://summer.iscas.ac.cn/#/org/prodetail/210370190?lang=en). In this report, the following two parts are covered: the first section is a basic introduction to the project, and the second section contains the implementation details of several multi-agent algorithms, followed by some workable usage examples.
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
        "publishedDate":"2021-09-29",
        "citationText":"Peter Chen, 2021"
    }"""

@def appendix = """
    ### Corrections
    If you see mistakes or want to suggest changes, please [create an issue](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues) in the source repository.
    """

@def bibliography = "bibliography.bib"

## 1. Project Information

Recent advances in reinforcement learning led to many breakthroughs in artificial intelligence. Some of the latest deep reinforcement learning algorithms have been implemented in [ReinforcementLearning.jl](https://juliareinforcementlearning.org/) with [Flux](https://fluxml.ai/). Currently, we only have some [CFR related algorithms](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningZoo/src/algorithms/cfr) implemented. We'd like to have more implemented, including **MADDPG**\dcite{DBLP:journals/corr/LoweWTHAM17}, **COMA**\dcite{DBLP:journals/corr/FoersterFANW17}, **NFSP**\dcite{DBLP:journals/corr/HeinrichS16}, **PSRO**\dcite{DBLP:journals/corr/abs-1909-12823}.

### 1.1 Schedule

| Date | Mission Content |
| :-----------: | :---------: |
| 07/01 -- 07/14 | Refer to the paper\dcite{DBLP:journals/corr/HeinrichS16} and the existing implementation to get familiar with the **NFSP** algorithm. |
| 07/15 -- 07/29 | Add **NFSP** algorithm into [ReinforcementLearningZoo.jl](https://juliareinforcementlearning.org/docs/rlzoo/), and test it on the [`KuhnPokerEnv`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.KuhnPokerEnv). |
| 07/30 -- 08/07 | Fix the existing bugs of **NFSP** and implement the **MADDPG** algorithm into ReinforcementLearningZoo.jl. |
| 08/08 -- 08/15 | Update the **MADDPG** algorithm and test it on the `KuhnPokerEnv`, also complete the **mid-term report**. |
| 08/16 -- 08/23 | Add support for environments of [`FULL_ACTION_SET`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.FULL_ACTION_SET) in **MADDPG** and test it on more games, such as [`simple_speaker_listener`](https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/scenarios/simple_speaker_listener.py). |
| 08/24 -- 08/30 | Fine-tuning the experiment [`MADDPG_SpeakerListener`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/481) and consider implementing **ED**\dcite{DBLP:journals/corr/abs-1903-05614} algorithm.|
| 08/31 -- 09/06 | Play games in 3rd party [`OpenSpiel`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.OpenSpielEnv) with **NFSP** algorithm. |
| 09/07 -- 09/13 | Implement **ED** algorithm and play "kuhn_poker" in `OpenSpiel` with **ED**. |
| 09/14 -- 09/20 | Fix the existing problems in the implemented **ED** algorithm and update the report. |
| 09/22 -- After Project | Complete the **final-term report**, and carry on maintaining the implemented algorithms. |

### 1.2 Accomplished Work

From July 1st to now, I have implemented the **Neural Fictitious Self-play(NFSP)**, **Multi-agent Deep Deterministic Policy Gradient(MADDPG)** and **Exploitability Descent(ED)** algorithms in [ReinforcementLearningZoo.jl](https://juliareinforcementlearning.org/docs/rlzoo/). Some workable experiments(see **Usage** part in each algorithm's section) are also added to the [documentation](https://juliareinforcementlearning.org/docs/experiments/). Besides, for testing the performance of **MADDPG** algorithm, I also implemented [`SpeakerListenerEnv`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.SpeakerListenerEnv-Tuple{}) in [ReinforcementLearningEnvironments.jl](https://juliareinforcementlearning.org/docs/rlenvs/). Related commits are listed below:

- [add Base.:(==) and Base.hash for AbstractEnv and test nash_conv on KuhnPokerEnv#348](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/348)
- [Supplement functions in ReservoirTrajectory and BehaviorCloningPolicy #390](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/390)
- [Implementation of NFSP and NFSP_KuhnPoker experiment #402](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/402)
- [correct nfsp implementation #439](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/439)
- [add MADDPG algorithm #444](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/444)
- [Update maddpg and the report #470](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/470)
- [Add the experiment of MADDPG. #481](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/481)
- [Update experiments of maddpg #487](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/487)
- [Play OpenSpiel envs with NFSP and try to add ED algorithm. #496](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/496)
- [Update ED algorithm and the report. #508](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/508)

## 2. Implementation and Usage

In this section, I will first briefly introduce some particular concepts in multi-agent reinforcement learning. Then I will review the [`Agent`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent) structure defined in [ReinforcementLearningCore.jl](https://juliareinforcementlearning.org/docs/rlcore/). After that,  I'll explain how these multi-agent algorithms(**NFSP**, **MADDPG**, and **ED**) are implemented, followed by some short examples to demonstrate how others can use them in their customized environments.

### 2.1 Terminology

This part is for introducing some terminologies in multi-agent reinforcement learning:

- [**Best Response**](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.BestResponsePolicy-Tuple{Any,%20Any,%20Any}):

Given a joint policy $\boldsymbol{\pi}$, which includes policies for all players, the **Best Response(BR)** policy for the player $i$ is the policy that achieves optimal payoff performance against $\boldsymbol{\pi}_{-i}$ :

$$
b_{i} \left(\boldsymbol{\pi}_{-i} \right) \in \mathrm{BR}\left(\boldsymbol{\pi}_{-i}\right)=\left\{\boldsymbol{\pi}_{i} \mid v_{i,\left(\boldsymbol{\pi}_{i}, \boldsymbol{\pi}_{-i}\right)}=\max _{\boldsymbol{\pi}_{i}^{\prime}} v_{i,\left(\boldsymbol{\pi}_{i}^{\prime}, \boldsymbol{\pi}_{-i}\right)}\right\}
$$

where $\boldsymbol{\pi}_{i}$ is the policy of the player $i$, $\boldsymbol{\pi}_{-i}$ refers to all policies in $\boldsymbol{\pi}$ except $\boldsymbol{\pi}_{i}$, and $v_{i,\left(\boldsymbol{\pi}_{i}^{\prime}, \boldsymbol{\pi}_{-i}\right)}$ is the expected reward of the joint policy $\left(\boldsymbol{\pi}_{i}^{\prime}, \boldsymbol{\pi}_{-i} \right)$ fot the player $i$.

- [**Nash Equilibrium**](https://en.wikipedia.org/wiki/Nash_equilibrium):

A **Nash Equilibrium** is a joint policy $\boldsymbol{\pi}$ such the each player's policy in $\boldsymbol{\pi}$ is a best reponse to the other policies. A common metric to measure the distance to **Nash Equilibrium** is [`nash_conv`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/src/ReinforcementLearningZoo/src/algorithms/cfr/nash_conv.jl#L29).

Given a joint policy $\boldsymbol{\pi}$, the **exploitability** for the player $i$ is the respective incentives to deviate from the current policy to the best response, denoted $\delta_{i}(\boldsymbol{\pi})=v_{i, \left(\boldsymbol{\pi}_{i}^{\prime}, \boldsymbol{\pi}_{-i}\right)} - v_{i, \boldsymbol{\pi}}$ where $\boldsymbol{\pi}_{i}^{\prime} \in \mathrm{BR}\left(\boldsymbol{\pi}_{-i}\right)$. In two-player [**zero-sum**](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.ZERO_SUM) games, an **$\epsilon$-Nash Equilibrium** policy is one where $\max _{i} \delta_{i}(\boldsymbol{\pi}) \leq \epsilon$. A **Nash Equilibrium** is achieved when $\epsilon = 0$. And the `nash_conv`$(\boldsymbol{\pi}) = \sum_{i} \delta_{i}\left(\boldsymbol{\pi}\right)$.

### 2.2 An Introduction to `Agent`

The [`Agent`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent) struct is an extended [`AbstractPolicy`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.AbstractPolicy) that includes a concrete policy and a trajectory. The trajectory is used to collect the necessary information to train the policy. In the existing [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningCore/src/policies/agents/agent.jl), the lifecycle of the [interactions](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent-Tuple{AbstractStage,%20AbstractEnv}) between agents and environments is split into several stages, including `PreEpisodeStage`, `PreActStage`, `PostActStage` and `PostEpisodeStage`.

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

### 2.3 Neural Fictitious Self-play(NFSP) algorithm

#### Brief Introduction

**Neural Fictitious Self-play(NFSP)**\dcite{DBLP:journals/corr/HeinrichS16} algorithm is a useful multi-agent algorithm that works well on imperfect-information games. Each agent who applies the **NFSP** algorithm has two inner agents, a **Reinforcement Learning (RL)** agent and a **Supervised Learning (SL)** agent. The **RL** agent is to find the best response to the state from the self-play process, and the **SL** agent is to learn the best response from the **RL** agent's policy. More importantly, **NFSP** also uses two technical innovations to ensure stability, including [**reservoir sampling**](https://en.wikipedia.org/wiki/Reservoir_sampling) for **SL** agent and **anticipatory dynamics**\dcite{1406126} when training. The following figure(from the paper\dcite{DBLP:journals/corr/abs-2104-10845}) shows the overall structure of **NFSP**(one agent).

\dfig{body;NFSP.png;The overall structure of **NFSP**(one agent).}

#### Implementation

In ReinforcementLearningZoo.jl, I implement the [`NFSPAgent`](https://juliareinforcementlearning.org/docs/rlzoo/#:~:text=ReinforcementLearningZoo.NFSPAgent) which defines the `NFSPAgent` struct and designs its behaviors according to the **NFSP** algorithm, including collecting needed information and how to update the policy. And the [`NFSPAgentManager`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.NFSPAgentManager) is a special multi-agent manager that all agents apply **NFSP** algorithm. Besides, in the [`abstract_nfsp`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/src/ReinforcementLearningZoo/src/algorithms/nfsp/abstract_nfsp.jl), I customize the `run` function for `NFSPAgentManager`.

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

Based on the setting [`stop_condition`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl#L126) and designed [`hook`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl#L15) in the experiment, you can just `run(nfsp, wrapped_env, stop_condition, hook)` to perform the experiment. Use [`Plots.plot`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/NFSP/JuliaRL_NFSP_KuhnPoker.jl#L136) to get the following result:

\dfig{body;JuliaRL_NFSP_KuhnPoker.png;Play KuhnPoker with NFSP.}

Besides, you can also play games implemented in 3rd party [`OpenSpiel`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.OpenSpielEnv)(see the [doc](https://openspiel.readthedocs.io/en/latest/julia.html)) with `NFSPAgentManager`, such as ["kuhn_poker"](https://openspiel.readthedocs.io/en/latest/games.html#kuhn-poker) and ["leduc_poker"](https://openspiel.readthedocs.io/en/latest/games.html#leduc-poker), just like the following:

```Julia
env = OpenSpielEnv("kuhn_poker")
wrapped_env = ActionTransformedEnv(
    env,
    # action is `0-based` in OpenSpiel, while `1-based` in Julia.
    action_mapping = a -> RLBase.current_player(env) == chance_player(env) ? a : Int(a - 1),
    action_space_mapping = as -> RLBase.current_player(env) == chance_player(env) ? 
        as : Base.OneTo(num_distinct_actions(env.game)),
)
# `InformationSet{String}()` is not supported when trainning.
wrapped_env = DefaultStateStyleEnv{InformationSet{Array}()}(wrapped_env)
```

Apart from the above environment wrapping, most details are the same with the experiment `JuliaRL_NFSP_KuhnPoker.` The result is shown below. For more details, you can refer to the [experiment](https://juliareinforcementlearning.org/docs/experiments/experiments/NFSP/JuliaRL_NFSP_OpenSpiel/#JuliaRL\\_NFSP\\_OpenSpiel(kuhn_poker)) `JuliaRL_NFSP_OpenSpiel(kuhn_poker)`.

\dfig{body;JuliaRL_NFSP_OpenSpiel(kuhn_poker).png;Play "kuhn_poker" in OpenSpiel with NFSP.}

### 2.4 Multi-agent Deep Deterministic Policy Gradient(MADDPG) algorithm

#### Brief Introduction

The **Multi-agent Deep Deterministic Policy Gradient(MADDPG)**\dcite{DBLP:journals/corr/LoweWTHAM17} algorithm improves the [Deep Deterministic Policy Gradient(DDPG)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), which also works well on multi-agent games. Based on the DDPG, the critic of each agent in **MADDPG** can get all agents' policies according to the paper\dcite{DBLP:journals/corr/LoweWTHAM17}'s hypothesis, including their personal states and actions, which can help to get a more reasonable score of the actor's policy. The following figure(from the paper\dcite{8846699}) illustrates the framework of **MADDPG**.

\dfig{body;MADDPG.png;The framework of **MADDPG**.}

#### Implementation

Given that the [`DDPGPolicy`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.DDPGPolicy-Tuple{}) is already implemented in the ReinforcementLearningZoo.jl, I implement the [`MADDPGManager`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.MADDPGManager) which is a special multi-agent manager that all agents apply `DDPGPolicy` with one **improved critic**. The structure of `MADDPGManager` is as the following:

```Julia
mutable struct MADDPGManager <: AbstractPolicy
    agents::Dict{<:Any, <:Agent}
    traces
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::AbstractRNG
end
```

Each agent in the `MADDPGManager` uses `DDPGPolicy` with one trajectory, which collects their own information. Note that the policy of the `Agent` should be wrapped with `NamedPolicy`. [`NamedPolicy`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.NamedPolicy) is a useful substruct of `AbstractPolicy` when meeting the multi-agent games, which combine the player's name and detailed policy. So that can use `Agent` 's [default behaviors](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.Agent-Tuple{AbstractStage,%20AbstractEnv}) to collect the necessary information.

As for updating the policy, the process is mainly the same as the [`DDPGPolicy`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/ddpg.jl#L127), apart from each agent's critic will assemble all agents' personal states and actions. For more details, you can refer to the [code](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/maddpg.jl#L64).

**Note that** when calculating the loss of actor's behavior network, we should add the `reg` term to improve the algorithm's performance, which differs from **DDPG**.

```Julia
gs2 = gradient(Flux.params(A)) do
    v = C(vcat(s, mu_actions)) |> vec
    reg = mean(A(batches[player][:state]) .^ 2)
    -mean(v) +  reg * 1e-3 # loss
end
```

#### Usage

Here `MADDPGManager` is used for the environments of [`SIMULTANEOUS`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.SIMULTANEOUS) and continuous action space(see the blog [Diagonal Gaussian Policies](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies)), or you can add an [action-related wrapper](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.ActionTransformedEnv-Tuple{Any}) to the environment to ensure it can work with the algorithm. There is one [experiment](https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker/#JuliaRL\\_MADDPG\\_KuhnPoker) `JuliaRL_MADDPG_KuhnPoker` as one usage example, which tests the algorithm on the Kuhn Poker game. Since the Kuhn Poker is one [`SEQUENTIAL`](ReinforcementLearningBase.SEQUENTIAL) game with discrete action space(see also the blog [Diagonal Gaussian Policies](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies)), I wrap the environment just like the following:

```Julia
wrapped_env = ActionTransformedEnv(
        StateTransformedEnv(
            env;
            state_mapping = s -> [findfirst(==(s), state_space(env))],
            state_space_mapping = ss -> [[findfirst(==(s), state_space(env))] for s in state_space(env)]
            ),
        ## drop the dummy action of the other agent.
        action_mapping = x -> length(x) == 1 ? x : Int(ceil(x[current_player(env)]) + 1),
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
    γ = 0.95f0,
    ρ = 0.99f0,
    na = na,
    start_steps = 1000,
    start_policy = RandomPolicy(-0.99..0.99; rng = rng),
    update_after = 1000,
    act_limit = 0.99,
    act_noise = 0.,
    rng = rng,
)
trajectory = CircularArraySARTTrajectory(
    capacity = 100_000, # replay buffer capacity
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
    SARTS, # trace's type
    512, # batch_size
    100, # update_freq
    0, # initial update_step
    rng
)
```

Plus on the [`stop_condition`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl#L111) and [`hook`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl#L15) in the experiment, you can also `run(agents, wrapped_env, stop_condition, hook)` to perform the experiment. Use [`Plots.scatter`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker.jl#L120) to get the following result:

\dfig{body;JuliaRL_MADDPG_KuhnPoker.png;Play KuhnPoker with MADDPG.}

However, `KuhnPoker` is not a good choice to show the performance of **MADDPG**. For testing the algorithm, I add [`SpeakerListenerEnv`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.SpeakerListenerEnv-Tuple{}) into [ReinforcementLearningEnvironments.jl](https://juliareinforcementlearning.org/docs/rlenvs), which is a simple cooperative multi-agent game.

Since two players have different dimensions of state and action in the `SpeakerListenerEnv`, the policy and the trajectory are customized as below:

```Julia
# initial the game.
env = SpeakerListenerEnv(max_steps = 25)
# network's parameter initialization method.
init = glorot_uniform(rng)
# critic's input units, including both players' states and actions.
critic_dim = sum(length(state(env, p)) + length(action_space(env, p)) for p in (:Speaker, :Listener))
# actor and critic's network structure.
create_actor(player) = Chain(
    Dense(length(state(env, player)), 64, relu; init = init),
    Dense(64, 64, relu; init = init),
    Dense(64, length(action_space(env, player)); init = init)
    )
create_critic(critic_dim) = Chain(
    Dense(critic_dim, 64, relu; init = init),
    Dense(64, 64, relu; init = init),
    Dense(64, 1; init = init),
    )
# concrete DDPGPolicy of the player.
create_policy(player) = DDPGPolicy(
    behavior_actor = NeuralNetworkApproximator(
        model = create_actor(player),
        optimizer = Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-2)),
    ),
    behavior_critic = NeuralNetworkApproximator(
        model = create_critic(critic_dim),
        optimizer = Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-2)),
    ),
    target_actor = NeuralNetworkApproximator(
        model = create_actor(player),
    ),
    target_critic = NeuralNetworkApproximator(
        model = create_critic(critic_dim),
    ),
    γ = 0.95f0,
    ρ = 0.99f0,
    na = length(action_space(env, player)),
    start_steps = 0,
    start_policy = nothing,
    update_after = 512 * env.max_steps, # batch_size * env.max_steps
    act_limit = 1.0,
    act_noise = 0.,
    )
create_trajectory(player) = CircularArraySARTTrajectory(
    capacity = 1_000_000, # replay buffer capacity
    state = Vector{Float64} => (length(state(env, player)), ),
    action = Vector{Float64} => (length(action_space(env, player)), ),
    )
```

Based on the above policy and trajectory, we can design the corresponding `MADDPGManager`:

```Julia
agents = MADDPGManager(
    Dict(
        player => Agent(
            policy = NamedPolicy(player, create_policy(player)),
            trajectory = create_trajectory(player),
        ) for player in (:Speaker, :Listener)
    ),
    SARTS, # trace's type
    512, # batch_size
    100, # update_freq
    0, # initial update_step
    rng
)
```

Add the [`stop_condition`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_SpeakerListener.jl#L108) and designed [`hook`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_SpeakerListener.jl#L15), we can simply `run(agents, env, stop_condition, hook)` to run the experiment and use [`Plots.plot`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_SpeakerListener.jl#L117) to get the following result.

\dfig{body;JuliaRL_MADDPG_SpeakerListenerEnv.png;Play SpeakerListenerEnv with MADDPG.}

### 2.5 Exploitability Descent(ED) algorithm

#### Brief Introduction

**Exploitability Descent(ED)**\dcite{DBLP:journals/corr/abs-1903-05614} is the algorithm to compute approximate equilibria in two-player [**zero-sum**](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.ZERO_SUM) [extensive-form games](https://en.wikipedia.org/wiki/Extensive-form_game) with imperfect information\dcite{osborne1994course}. The **ED** algorithm directly optimizes the player's policy against the worst case oppoent. The **exploitability** of each player applying **ED**'s policy converges asymptotically to zero. Hence in self-play, the joint policy $\boldsymbol{\pi}$ converges to an approximate **Nash Equilibrium**.

#### Implementation

Unlike the above two algorithms, the **ED** algorithm does not need to collect the information in each stage. Instead, on each iteration, there are the following two steps that occur for each player employing the **ED** algorithm:

- Compute the **best response** policy to each player's policy;
- Perform the **gradient ascent** on the policy to increase each player's utility against the respective best responder(i.e. the opponent), which aims to decrease each player's **exploitability**.

In ReinforcementLearingZoo.jl, I implement [`EDPolicy`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.EDPolicy) which defines the `EDPolicy` struct and customize the interactions with the environments:

```Julia
## definition
mutable struct EDPolicy{P<:NeuralNetworkApproximator, E<:AbstractExplorer}
    opponent::Any # record the opponent's name.
    learner::P # get the action value of the state.
    explorer::E # by default use `WeightedSoftmaxExplorer`.
end
## interactions with the environment
function (π::EDPolicy)(env::AbstractEnv)
    s = state(env)
    s = send_to_device(device(π.learner), Flux.unsqueeze(s, ndims(s) + 1))
    logits = π.learner(s) |> vec |> send_to_host
    ActionStyle(env) isa MinimalActionSet ? π.explorer(logits) : 
        π.explorer(logits, legal_action_space_mask(env))
end
# set the `_device` function for convenience transferring the variable to the corresponding device.
_device(π::EDPolicy, x) = send_to_device(device(π.learner), x)

function RLBase.prob(π::EDPolicy, env::AbstractEnv; to_host::Bool = true)
    s = @ignore state(env) |> x -> Flux.unsqueeze(x, ndims(x) + 1) |> x -> _device(π, x)
    logits = π.learner(s) |> vec
    mask = @ignore legal_action_space_mask(env) |> x -> _device(π, x)
    p = ActionStyle(env) isa MinimalActionSet ? prob(π.explorer, logits) : prob(π.explorer, logits, mask)
    to_host ? p |> send_to_host : p
end

function RLBase.prob(π::EDPolicy, env::AbstractEnv, action)
    A = action_space(env)
    P = prob(π, env)
    @assert length(A) == length(P)
    if A isa Base.OneTo
        P[action]
    else
        for (a, p) in zip(A, P)
            if a == action
                return p
            end
        end
        @error "action[$action] is not found in action space[$(action_space(env))]"
    end
end
```

Here I use many macro operators [`@ignore`](https://fluxml.ai/Zygote.jl/latest/utils/#Zygote.ignore) for being able to compute the gradient of the parameters. Also, I design the [`update!`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/src/ReinforcementLearningZoo/src/algorithms/exploitability_descent/EDPolicy.jl#L73) function for `EDPolicy` when getting the opponent's **best response** policy:

```Julia
function RLBase.update!(
    π::EDPolicy, 
    Opponent_BR::BestResponsePolicy, 
    env::AbstractEnv,
    player::Any,
)
    reset!(env)

    # construct policy vs best response
    policy_vs_br = PolicyVsBestReponse(
        MultiAgentManager(
            NamedPolicy(player, π),
            NamedPolicy(π.opponent, Opponent_BR),
            ),
        env,
        player,
    )
    info_states = collect(keys(policy_vs_br.info_reach_prob))
    cfr_reach_prob = collect(values(policy_vs_br.info_reach_prob)) |> x -> _device(π, x)

    gs = gradient(Flux.params(π.learner)) do
        # Vector of shape `(length(info_states), 1)`
        # compute expected reward from the start of `e` with policy_vs_best_reponse
        # baseline = ∑ₐ πᵢ(s, a) * q(s, a)
        baseline = @ignore Flux.stack(([values_vs_br(policy_vs_br, e)] for e in info_states), 1) |> x -> _device(π, x)
        
        # Vector of shape `(length(info_states), length(action_space))`
        # compute expected reward from the start of `e` when playing each action.
        q_values = Flux.stack((q_value(π, policy_vs_br, e) for e in info_states), 1)

        advantage = q_values .- baseline
        # Vector of shape `(length(info_states), length(action_space))`
        # get the prob of each action with `e`, i.e., πᵢ(s, a).
        policy_values = Flux.stack((prob(π, e, to_host = false) for e in info_states), 1)

        # get each info_state's loss
        # ∑ₐ πᵢ(s, a) * (q(s, a) - baseline), where baseline = ∑ₐ πᵢ(s, a) * q(s, a).
        loss_per_state = - sum(policy_values .* advantage, dims = 2)

        sum(loss_per_state .* cfr_reach_prob)
    end
    update!(π.learner, gs)
end
```

Here I implement one [`PolicyVsBestResponse`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/src/ReinforcementLearningZoo/src/algorithms/exploitability_descent/EDPolicy.jl#L118) struct for computing related values, such as the [probabilities](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/src/ReinforcementLearningZoo/src/algorithms/exploitability_descent/EDPolicy.jl#L140) of opponent's reaching one particular environment in playing, and the [expected reward](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/src/ReinforcementLearningZoo/src/algorithms/exploitability_descent/EDPolicy.jl#L161) from the start of a specific environment when against the opponent's **best response**.

Besides, I implement the [`EDManager`](https://juliareinforcementlearning.org/docs/rlzoo/#ReinforcementLearningZoo.EDManager), which is a special multi-agent manager that all agents utilize the **ED** algorithm, and set the particular [`run`](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/3851546ec2ce529a490bb5dacc1b6e0ddaaea941/src/ReinforcementLearningZoo/src/algorithms/exploitability_descent/exploitability_descent.jl#L27) function for running the experiment:

```Julia
## run function
function Base.run(
    π::EDManager,
    env::AbstractEnv,
    stop_condition = StopAfterEpisode(1),
    hook::AbstractHook = EmptyHook(),
)
    @assert NumAgentStyle(env) == MultiAgent(2) "ED algorithm only support 2-players games."
    @assert UtilityStyle(env) isa ZeroSum "ED algorithm only support zero-sum games."

    is_stop = false

    while !is_stop
        RLBase.reset!(env)
        hook(PRE_EPISODE_STAGE, π, env)

        for (player, policy) in π.agents
            # construct opponent's best response policy.
            oppo_best_response = BestResponsePolicy(π, env, policy.opponent)
            # update player's policy by using policy-gradient.
            update!(policy, oppo_best_response, env, player)
        end

        # run one episode for update stop_condition
        RLBase.reset!(env)
        while !is_terminated(env)
            π(env) |> env
        end

        if stop_condition(π, env)
            is_stop = true
            break
        end
        hook(POST_EPISODE_STAGE, π, env)
    end
    hook(POST_EXPERIMENT_STAGE, π, env)
    hook
end
```

#### Usage

According to the paper\dcite{DBLP:journals/corr/abs-1903-05614}, `EDmanager` only supports for the two-player **zero-sum** games. There is one [experiment](https://juliareinforcementlearning.org/docs/experiments/experiments/ED/JuliaRL_ED_OpenSpiel/#JuliaRL\\_ED\\_OpenSpiel(kuhn_poker)) `JuliaRL_ED_OpenSpiel` as one usage example, which tests the algorithm on the Kuhn Poker game in 3rd-party `OpenSpiel`. Here I also customized the `hook` and `stop_condition` for testing the implemented **ED** algorithm.

New `hook` is designed as the following:

```Julia
mutable struct KuhnOpenNewEDHook <: AbstractHook
    episode::Int
    eval_freq::Int
    episodes::Vector{Int}
    results::Vector{Float64}
end

function (hook::KuhnOpenNewEDHook)(::PreEpisodeStage, policy, env)
    hook.episode += 1
    if hook.episode % hook.eval_freq == 1
        push!(hook.episodes, hook.episode)
        ## get nash_conv of the current policy.
        push!(hook.results, RLZoo.nash_conv(policy, env))
    end

    ## update agents' learning rate.
    for (_, agent) in policy.agents
        agent.learner.optimizer[2].eta = 1.0 / sqrt(hook.episode)
    end
end
```

Next, wrap the environment and initialize the `EDmanager`, `hook` and `stop_condition`:

```Julia
# set random seed.
rng = StableRNG(123)
# wrap and initial the OpenSpiel environment.
env = OpenSpielEnv(game)
wrapped_env = ActionTransformedEnv(
    env,
    action_mapping = a -> RLBase.current_player(env) == chance_player(env) ? a : Int(a - 1),
    action_space_mapping = as -> RLBase.current_player(env) == chance_player(env) ? 
        as : Base.OneTo(num_distinct_actions(env.game)),
)
wrapped_env = DefaultStateStyleEnv{InformationSet{Array}()}(wrapped_env)
player = 0 # or 1
ns, na = length(state(wrapped_env, player)), length(action_space(wrapped_env, player))
# construct the `EDmanager`.
create_network() = Chain(
    Dense(ns, 64, relu;init = glorot_uniform(rng)),
    Dense(64, na;init = glorot_uniform(rng))
)
create_learner() = NeuralNetworkApproximator(
    model = create_network(),
    # set the l2-regularization and use gradient descent optimizer.
    optimizer = Flux.Optimise.Optimiser(WeightDecay(0.001), Descent())
)
EDmanager = EDManager(
    Dict(
        player => EDPolicy(
            1 - player, # opponent
            create_learner(), # neural network learner
            WeightedSoftmaxExplorer(), # explorer
        ) for player in players(env) if player != chance_player(env)
    )
)
# initialize the `stop_condition` and `hook`.
stop_condition = StopAfterEpisode(100_000, is_show_progress=!haskey(ENV, "CI"))
hook = KuhnOpenNewEDHook(0, 100, [], [])
```

Based on the above setting, you can perform the experiment by `run(EDmanager, wrapped_env, stop_condition, hook)`. Use the following `Plots.plot` to get the experiment's result:

```Julia
plot(hook.episodes, hook.results, scale=:log10, xlabel="episode", ylabel="nash_conv")
```

\dfig{body;JuliaRL_ED_OpenSpiel(kuhn_poker).png;Play "kuhn_poker" in OpenSpiel with ED.}
