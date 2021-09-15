export NFSPAgent

## definition
"""
    NFSPAgent(; rl_agent::Agent, sl_agent::Agent, args...)

Neural Fictitious Self-Play (NFSP) agent implemented in Julia. 
See the paper https://arxiv.org/abs/1603.01121 for more details.

# Keyword arguments

- `rl_agent::Agent`, Reinforcement Learning(RL) agent(default `QBasedPolicy` here, use `DQN` for example), which works to search the best response from the self-play process.
- `sl_agent::Agent`, Supervisor Learning(SL) agent(use `BehaviorCloningPolicy` for example), which works to learn the best response from the rl_agent's policy.
- `η`, anticipatory parameter, the probability to use `ϵ-greedy(Q)` policy when training the agent.
- `rng=Random.GLOBAL_RNG`.
- `update_freq::Int`: the frequency of updating the agents' `approximator`.
- `update_step::Int`, count the step.
- `mode::Bool`, used when learning, true as BestResponse(rl_agent's output), false as AveragePolicy(sl_agent's output).
"""
mutable struct NFSPAgent <: AbstractPolicy
    rl_agent::Agent
    sl_agent::Agent
    η
    rng
    update_freq::Int
    update_step::Int
    mode::Bool
end

## interactions when evaluation.
(π::NFSPAgent)(env::AbstractEnv) = π.sl_agent(env)

RLBase.prob(π::NFSPAgent, env::AbstractEnv, args...) = prob(π.sl_agent.policy, env, args...)

## update nfsp(also the env) when training.
function RLBase.update!(π::NFSPAgent, env::AbstractEnv)
    player = current_player(env)
    action = π.mode ? π.rl_agent(env) : π.sl_agent(env)
    π(PRE_ACT_STAGE, env, action)
    env(action)
    π(POST_ACT_STAGE, env, player)
end

function (π::NFSPAgent)(stage::PreEpisodeStage, env::AbstractEnv, ::Any)
    # delete the terminal state and dummy action.
    update!(π.rl_agent.trajectory, π.rl_agent.policy, env, stage)

    # set the train's mode before the episode.
    π.mode = rand(π.rng) < π.η
end

function (π::NFSPAgent)(stage::PreActStage, env::AbstractEnv, action)
    rl = π.rl_agent
    sl = π.sl_agent

    # update trajectory
    if π.mode
        update!(sl.trajectory, sl.policy, env, stage, action)
        rl(stage, env, action)# also update rl_policy(both learner network and target network).
    else
        update!(rl.trajectory, rl.policy, env, stage, action)
    end

    # update policy
    π.update_step += 1
    if π.update_step % π.update_freq == 0
        if π.mode
            update!(sl.policy, sl.trajectory)
        else
            rl_learn!(rl.policy, rl.trajectory) # only update rl_policy's learner.
            update!(sl.policy, sl.trajectory)
        end
    end
end

function (π::NFSPAgent)(::PostActStage, env::AbstractEnv, player::Any)
    push!(π.rl_agent.trajectory[:reward], reward(env, player))
    push!(π.rl_agent.trajectory[:terminal], is_terminated(env))
end

function (π::NFSPAgent)(::PostEpisodeStage, env::AbstractEnv, player::Any)
    rl = π.rl_agent
    sl = π.sl_agent

    # update trajectory
    # Note that for the `TERMINAL_REWARD` and `SEQUENTIAL` games, some players may not record their real reward and terminated judgment.
    if !rl.trajectory[:terminal][end]
        rl.trajectory[:reward][end] = reward(env, player)
        rl.trajectory[:terminal][end] = is_terminated(env)
    end

    # collect state and dummy action to rl.trajectory
    action = rand(action_space(env, player))
    push!(rl.trajectory[:state], state(env, player))
    push!(rl.trajectory[:action], action)
    if haskey(rl.trajectory, :legal_actions_mask)
        push!(rl.trajectory[:legal_actions_mask], legal_action_space_mask(env, player))
    end
    
    # update the policy    
    π.update_step += 1
    if π.update_step % π.update_freq == 0
        if π.mode
            update!(sl.policy, sl.trajectory)
        else
            rl_learn!(rl.policy, rl.trajectory) # only update rl_policy's learner.
            update!(sl.policy, sl.trajectory)
        end
    end
end

# here just update the rl's approximator, not update target_approximator.
function rl_learn!(policy::QBasedPolicy, t::AbstractTrajectory)
    learner = policy.learner
    length(t[:terminal]) - learner.sampler.n <= learner.min_replay_history && return
    
    _, batch = sample(learner.rng, t, learner.sampler)

    if t isa PrioritizedTrajectory
        priorities = update!(learner, batch)
        t[:priority][inds] .= priorities
    else
        update!(learner, batch)
    end
end
