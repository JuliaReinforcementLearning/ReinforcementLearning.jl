#####
# some common components for Prioritized Experience Replay based methods
#####

const PERLearners = Union{PrioritizedDQNLearner, RainbowLearner, IQNLearner}

function extract_experience(t::AbstractTrajectory, learner::PERLearners)
    s = learner.stack_size
    h = learner.update_horizon
    n = learner.batch_size
    γ = learner.γ

    # 1. sample indices based on priority
    valid_ind_range = isnothing(s) ? (1:(length(t)-h)) : (s:(length(t)-h))
    if t isa CircularCompactPSARTSATrajectory
        inds = Vector{Int}(undef, n)
        priorities = Vector{Float32}(undef, n)
        for i in 1:n
            ind, p = sample(learner.rng, get_trace(t, :priority))
            while ind ∉ valid_ind_range
                ind, p = sample(learner.rng, get_trace(t, :priority))
            end
            inds[i] = ind
            priorities[i] = p
        end
    else
        inds = rand(learner.rng, valid_ind_range, n)
        priorities = nothing
    end

    # 2. extract SARTS
    states = consecutive_view(get_trace(t, :state), inds; n_stack = s)
    actions = consecutive_view(get_trace(t, :action), inds)
    next_states = consecutive_view(get_trace(t, :state), inds .+ h; n_stack = s)
    consecutive_rewards = consecutive_view(get_trace(t, :reward), inds; n_horizon = h)
    consecutive_terminals = consecutive_view(get_trace(t, :terminal), inds; n_horizon = h)
    rewards, terminals = zeros(Float32, n), fill(false, n)

    rewards = discount_rewards_reduced(
        consecutive_rewards,
        γ;
        terminal = consecutive_terminals,
        dims = 1,
    )
    terminals = mapslices(any, consecutive_terminals; dims = 1) |> vec

    inds,
    (
        states = states,
        actions = actions,
        rewards = rewards,
        terminals = terminals,
        next_states = next_states,
        priorities = priorities
    )
end

function RLBase.update!(p::QBasedPolicy{<:PERLearners}, t::AbstractTrajectory)
    learner = p.learner
    length(t) < learner.min_replay_history && return

    learner.update_step += 1

    if learner.update_step % learner.target_update_freq == 0
        copyto!(learner.target_approximator, learner.approximator)
    end

    learner.update_step % learner.update_freq == 0 || return

    inds, experience = extract_experience(t, p.learner)

    if t isa CircularCompactPSARTSATrajectory
        priorities = update!(p.learner, experience)
        get_trace(t, :priority)[inds] .= priorities
    else
        update!(p.learner, experience)
    end
end

function (
    agent::Agent{
        <:QBasedPolicy{<:PERLearners},
        <:CircularCompactPSARTSATrajectory,
    }
)(
    ::RLCore.Training{PostActStage},
    obs,
)
    push!(
        agent.trajectory;
        reward = get_reward(obs),
        terminal = get_terminal(obs),
        priority = agent.policy.learner.default_priority,
    )
    nothing
end