export OfflinePolicy, JuliaRLTransition, gen_JuliaRL_dataset

export calculate_CQL_loss, maximum_mean_discrepancy_loss

struct JuliaRLTransition
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
        for _ in 1:l.pretrain_step
            inds, batch = sample(l.rng, p.dataset, p.batch_size)
            update!(l, batch)
        end
    end
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

"""
    gen_JuliaRL_dataset(alg::Symbol, env::Symbol, type::AbstractString; dataset_size)

Generate the dataset by trajectory from the trajectory obtained from the experiment (`alg` + `env`). `type` represents the method of collecting data. Possible values: random/medium/expert. `dataset_size` is the size of the generated dataset.
"""
function gen_JuliaRL_dataset(alg::Symbol, env::Symbol, type::AbstractString; dataset_size::Int)
    dataset_ex = Experiment(
            Val(:GenDataset),
            Val(alg),
            Val(env),
            type;
            dataset_size = dataset_size)
    
    run(dataset_ex)

    dataset = []
    s, a, r, t = dataset_ex.policy.trajectory.traces
    for i in 1:dataset_size
        push!(dataset, JuliaRLTransition(s[:, i], a[i], r[i], t[i], s[:, i+1]))
    end
    dataset
end

function StatsBase.sample(rng::AbstractRNG, dataset::Vector{T}, batch_size::Int) where {T}
    valid_range = 1:length(dataset)
    inds = rand(rng, valid_range, batch_size)
    batch_data = dataset[inds]
    s_length = size(batch_data[1].state)[1]
    a_type = typeof(batch_data[1].action)

    s = Array{Float32}(undef, s_length, batch_size)
    s′ = Array{Float32}(undef, s_length, batch_size)
    a = Array{a_type}(undef, batch_size)
    r = Array{Float32}(undef, batch_size)
    t = Array{Float32}(undef, batch_size)
    for (i, data) in enumerate(batch_data)
        s[:, i] = data.state
        a[i] = data.action
        s′[:, i] = data.next_state
        r[i] = data.reward
        t[i] = data.terminal
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
