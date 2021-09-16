export OfflinePolicy, AtariRLTransition

export calculate_CQL_loss, maximum_mean_discrepancy_loss

using ProfileView

struct AtariRLTransition
    state
    action
    reward
    terminal
    next_state
end

Base.@kwdef struct OfflinePolicy{L,T} <: AbstractPolicy
    learner::L
    dataset::T
    continuous::Bool
    batch_size::Int
end

(π::OfflinePolicy)(env) = π(env, ActionStyle(env), action_space(env))

function (π::OfflinePolicy)(env, ::MinimalActionSet, ::Base.OneTo)
    if π.continuous
        π.learner(env)
    else
        findmax(π.learner(env))[2]
    end
end
(π::OfflinePolicy)(env, ::FullActionSet, ::Base.OneTo) = findmax(π.learner(env), legal_action_space_mask(env))[2]

function (π::OfflinePolicy)(env, ::MinimalActionSet, A)
    if π.continuous
        π.learner(env)
    else
        A[findmax(π.learner(env))[2]]
    end
end
(π::OfflinePolicy)(env, ::FullActionSet, A) = A[findmax(π.learner(env), legal_action_space_mask(env))[2]]

function RLBase.update!(
    p::OfflinePolicy,
    traj::AbstractTrajectory,
    ::AbstractEnv,
    ::PreExperimentStage,
)
    l = p.learner
    if in(:pretrain_step, fieldnames(typeof(l)))
        println("Pretrain...")
        @profview for _ in 1:l.pretrain_step
            #inds, batch = sample(l.rng, p.dataset, p.batch_size)
            batch_data = take!(p.dataset)
            batch = NamedTuple{SARTS}((batch_data.state, batch_data.action, batch_data.reward, batch_data.terminal, batch_data.next_state))
            update!(l, batch)
        end
    end
    Profile.print()
    @assert 1 == 2
end

function RLBase.update!(
    p::OfflinePolicy,
    traj::AbstractTrajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    l = p.learner
    l.update_step += 1

    if in(:target_update_freq, fieldnames(typeof(l))) && l.update_step % l.target_update_freq == 0
        copyto!(l.target_approximator, l.approximator)
    end

    l.update_step % l.update_freq == 0 || return

    inds, batch = sample(l.rng, p.dataset, p.batch_size)

    update!(l, batch)
end

function StatsBase.sample(rng::AbstractRNG, dataset::Vector{T}, batch_size::Int) where {T}
    valid_range = 1:length(dataset)
    inds = rand(rng, valid_range, batch_size)
    batch_data = dataset[inds]
    s_length = size(batch_data[1].state)[1]

    s = Array{Float32}(undef, s_length, batch_size)
    s′ = Array{Float32}(undef, s_length, batch_size)
    a = []
    r = []
    t = []
    for (i, data) in enumerate(batch_data)
        s[:, i] = data.state
        push!(a, data.action)
        s′[:, i] = data.next_state
        push!(r, data.reward)
        push!(t, data.terminal)
    end
    batch = NamedTuple{SARTS}((s, a, r, t, s′))
    inds, batch
end

"""
    calculate_CQL_loss(q_value, action; method)
See paper: [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
"""
function calculate_CQL_loss(q_value::Matrix{T}, action::Vector{R}; method = "CQL(H)") where {T, R}
    if method == "CQL(H)"
        cql_loss = mean(log.(sum(exp.(q_value), dims=1)) .- q_value[action])
    else
        @error Wrong method parameter
    end
    return cql_loss
end

function maximum_mean_discrepancy_loss(raw_sample_action, raw_actor_action, type::Symbol, mmd_σ::Float32=10.0f0)
    A, B, N = size(raw_sample_action)
    diff_xx = reshape(raw_sample_action, A, B, N, 1) .- reshape(raw_sample_action, A, B, 1, N)
    diff_xy = reshape(raw_sample_action, A, B, N, 1) .- reshape(raw_actor_action, A, B, 1, N)
    diff_yy = reshape(raw_actor_action, A, B, N, 1) .- reshape(raw_actor_action, A, B, 1, N)
    diff_xx = calculate_sample_distance(diff_xx, type, mmd_σ)
    diff_xy = calculate_sample_distance(diff_xy, type, mmd_σ)
    diff_yy = calculate_sample_distance(diff_yy, type, mmd_σ)
    mmd_loss = sqrt.(diff_xx .+ diff_yy .- 2.0f0 .* diff_xy .+ 1.0f-6)
end

function calculate_sample_distance(diff, type::Symbol, mmd_σ::Float32)
    if type == :gaussian
        diff = diff .^ 2
    elseif type == :laplacian
        diff = abs.(diff)
    else
        error("Wrong parameter.")
    end
    return vec(mean(exp.(-sum(diff, dims=1) ./ (2.0f0 * mmd_σ)), dims=(3, 4)))
end
