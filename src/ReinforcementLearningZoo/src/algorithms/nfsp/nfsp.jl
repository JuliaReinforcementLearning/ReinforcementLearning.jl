export NFSPAgent


"""
    NFSPAgent(; rl_agent::Agent, sl_agent::Agent, args...)

Neural Fictitious Self-Play (NFSP) agent implemented in Julia. 
See the paper https://arxiv.org/abs/1603.01121 for more details.

# Keyword arguments

- `rl_agent::Agent`, Reinforcement Learning(RL) agent(use `DQN` for example), which works to search the best response from the self-play process.
- `sl_agent::Agent`, Supervisor Learning(SL) agent(use `BehaviorCloningPolicy` for example), which works to learn the best response from the rl_agent's policy.
- `η`, anticipatory parameter, the probability to use `ϵ-greedy(Q)` policy when training the agent.
- `rng=Random.GLOBAL_RNG`.
- `update_freq::Int`: the frequency of updating the agents' `approximator`.
- `step_counter::Int`, count the step.
- `mode::Bool`, used when learning, true as BestResponse(rl_agent's output), false as AveragePolicy(sl_agent's output).
"""
mutable struct NFSPAgent <: AbstractPolicy
    rl_agent::Agent
    sl_agent::Agent
    η
    rng
    update_freq::Int
    step_counter::Int
    mode::Bool
end

function RLBase.update!(π::NFSPAgent, env::AbstractEnv)
    action = π.mode ? π.rl_agent(env) : π.sl_agent(env)
    π(PRE_ACT_STAGE, env, action)
    env(action)
    π(POST_ACT_STAGE, env)
end

RLBase.prob(π::NFSPAgent, env::AbstractEnv, args...) = prob(π.sl_agent.policy, env, args...)

function (π::NFSPAgent)(stage::PreActStage, env::AbstractEnv, action)
    rl = π.rl_agent
    sl = π.sl_agent

    # update trajectory
    if π.mode
        action_probs = prob(rl.policy, env)
        if typeof(action_probs) == Categorical{Float64, Vector{Float64}}
            action_probs = probs(action_probs)
        end

        RLBase.update!(sl.trajectory, sl.policy, env, stage, action_probs)
        rl(PRE_ACT_STAGE, env, action) # also update rl_agent's network
    else
        RLBase.update!(rl.trajectory, rl.policy, env, stage, action)
    end
    
    # update agent's approximator
    π.step_counter += 1
    if π.step_counter % π.update_freq == 0
        RLBase.update!(sl.policy, sl.trajectory)
        if !π.mode
            rl_learn(π.rl_agent)
        end
    end
end

(π::NFSPAgent)(stage::PostActStage, env::AbstractEnv) = π.rl_agent(stage, env)

function (π::NFSPAgent)(stage::PostEpisodeStage, env::AbstractEnv)
    rl = π.rl_agent
    sl = π.sl_agent
    RLBase.update!(rl.trajectory, rl.policy, env, stage)
    
    # train the agent
    π.step_counter += 1
    if π.step_counter % π.update_freq == 0
        RLBase.update!(sl.policy, sl.trajectory)
        if !π.mode
            rl_learn(π.rl_agent)
        end
    end
end

# Following is the supplement functions
# if the implementation work well, following function maybe move to the correspond file.
function rl_learn(rl_agent::Agent{<:QBasedPolicy, <:AbstractTrajectory})
    # just learn the approximator, not update target_approximator
    learner, t = rl_agent.policy.learner, rl_agent.trajectory
    length(t[:terminal]) - learner.sampler.n <= learner.min_replay_history && return
    
    _, batch = sample(learner.rng, t, learner.sampler)

    if t isa PrioritizedTrajectory
        priorities = update!(learner, batch)
        t[:priority][inds] .= priorities
    else
        update!(learner, batch)
    end
end

function RLBase.update!(p::BehaviorCloningPolicy, batch::NamedTuple{(:state, :action_probs)})
    s, probs = batch.state, batch.action_probs
    m = p.approximator
    gs = gradient(params(m)) do
        ŷ = m(s)
        y = probs
        Flux.Losses.logitcrossentropy(ŷ, y)
    end
    update!(m, gs)
end

function RLBase.update!(
    trajectory::ReservoirTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action_probs::Vector{Float64},
)
    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)
    if haskey(trajectory.buffer, :legal_actions_mask)
        lasm =
            policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
            legal_action_space_mask(env)
        push!(trajectory; :state => s, :action_probs => action_probs, :legal_actions_mask => lasm)
    else
        push!(trajectory; :state => s, :action_probs => action_probs)
    end
end
