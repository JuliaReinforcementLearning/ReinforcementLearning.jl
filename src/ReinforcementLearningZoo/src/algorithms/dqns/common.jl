#####
# some common components for Prioritized Experience Replay based methods
#####

const PERLearners = Union{PrioritizedDQNLearner,RainbowLearner,IQNLearner}

function extract_experience(t::AbstractTrajectory, learner::PERLearners)
    s = learner.stack_size
    h = learner.update_horizon
    n = learner.batch_size
    γ = learner.γ

    # 1. sample indices based on priority
    valid_ind_range =
        isnothing(s) ? (1:(length(t[:terminal])-h)) : (s:(length(t[:terminal])-h))
    if haskey(t, :priority)
        inds = Vector{Int}(undef, n)
        priorities = Vector{Float32}(undef, n)
        for i in 1:n
            ind, p = sample(learner.rng, t[:priority])
            while ind ∉ valid_ind_range
                ind, p = sample(learner.rng, t[:priority])
            end
            inds[i] = ind
            priorities[i] = p
        end
    else
        inds = rand(learner.rng, valid_ind_range, n)
        priorities = nothing
    end

    next_inds = inds .+ h

    # 2. extract SARTS
    states = consecutive_view(t[:state], inds; n_stack = s)
    actions = consecutive_view(t[:action], inds)
    next_states = consecutive_view(t[:state], next_inds; n_stack = s)

    if haskey(t, :legal_actions_mask)
        legal_actions_mask = consecutive_view(t[:legal_actions_mask], inds)
        next_legal_actions_mask = consecutive_view(t[:next_legal_actions_mask], inds)
    else
        legal_actions_mask = nothing
        next_legal_actions_mask = nothing
    end

    consecutive_rewards = consecutive_view(t[:reward], inds; n_horizon = h)
    consecutive_terminals = consecutive_view(t[:terminal], inds; n_horizon = h)
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
        legal_actions_mask = legal_actions_mask,
        actions = actions,
        rewards = rewards,
        terminals = terminals,
        next_states = next_states,
        next_legal_actions_mask = next_legal_actions_mask,
        priorities = priorities,
    )
end

function RLBase.update!(p::QBasedPolicy{<:PERLearners}, t::AbstractTrajectory)
    learner = p.learner
    length(t[:terminal]) < learner.min_replay_history && return

    learner.update_step += 1

    if learner.update_step % learner.target_update_freq == 0
        copyto!(learner.target_approximator, learner.approximator)
    end

    learner.update_step % learner.update_freq == 0 || return

    inds, experience = extract_experience(t, p.learner)

    if haskey(t, :priority)
        priorities = update!(p.learner, experience)
        t[:priority][inds] .= priorities
    else
        update!(p.learner, experience)
    end
end

function (agent::Agent{<:QBasedPolicy{<:PERLearners}})(
    ::RLCore.Training{PostActStage},
    env,
)
    push!(
        agent.trajectory;
        reward = get_reward(env),
        terminal = get_terminal(env),
    )
    if haskey(agent.trajectory, :priority)
        push!(agent.trajectory; priority = agent.policy.learner.default_priority)
    end
    nothing
end
