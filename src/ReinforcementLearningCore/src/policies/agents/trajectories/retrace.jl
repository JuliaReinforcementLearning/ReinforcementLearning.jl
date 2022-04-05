export RetraceTrajectory

mutable struct RetraceTrajectory{T, S <: AbstractSampler} <: AbstractTrajectory
    traces::T
    λ::Float32
    batch_sampler::S
end

function RetraceTrajectory(;capacity, batch_size, λ = 0.9f0, nsteps, stack_size = nothing, action_log_prob, rng = Random.GLOBAL_RNG, kwargs...)
    traj = merge(
        CircularArrayTrajectory(;
            capacity = capacity + 1,
            action_log_prob = action_log_prob,
        ),
        CircularArraySARTTrajectory(; capacity = capacity, kwargs...),
    )
    sampler = NStepBatchSampler(γ = 1f0, n= nsteps, batch_size = batch_size, stack_size = stack_size, rng = rng)
    RetraceTrajectory(traj.traces, λ, sampler)
end

function fetch!(sampler::AbstractSampler, traj::RetraceTrajectory, inds::Vector{Int})
    n, bz, sz = sampler.γ, sampler.n, sampler.batch_size, sampler.stack_size
    
    #with retrace we need all states and actions from inds to inds .+ n 
    #Am I right to say that cache is not needed here ?
    s = consecutive_view(traj[:state], inds; n_stack = sz, n_horizon = n) #last dim is batch_size, second to last dim is n
    a = consecutive_view(traj[:action], inds, n_horizon = n)
    r = consecutive_view(traj[:reward], inds; n_horizon = n)
    t = consecutive_view(traj[:terminal], inds; n_horizon = n)
    logpi = consecutive_view(traj[:action_log_prob], inds; n_horizon = n)
    
    k = fill(n, bz) #stores the offset of the terminal/last step of each sample. This will be used to truncate the trajectory when computing the targets. 
    # make sure that we only consider experiences in current episode
    for i in 1:bz
        m = findfirst(view(consecutive_terminals, :, i))
        if !isnothing(m) #if isnothing, then we use all n sampled steps
            k[i] = m
        end
    end

    batch = NamedTuple{RetraceTraces}((s, a, r, t, logpi))

    if isnothing(sampler.cache)
        sampler.cache = map(batch) do x
            convert(Array, x)
        end
    else
        map(sampler.cache, batch) do dest, src
            copyto!(dest, src)
        end
    end
    return sampler.cache, k
end

function q_targets(p::AbstractPolicy, traj::RetraceTrajectory, qnetwork::QNetwork, batch, k)
    states, actions, rewards, terminals, logμs = batch
    s = send_to_device(device(p), states) 
    a = send_to_device(device(p), actions)
    a_t = p(p.rng, s, is_sampling = true)
    qs_t = qnetwork.target(s, a_t) |> send_to_host #assume qnetwork handles state and action as it is.
    qs = qnetwork(s, a) |> send_to_host
    logπs = p(s, a) |> send_to_host
    ci(logπ, logμ) = traj.λ*min(1, exp(logπ)/exp(logμ))
    targets = zeros(Float32, size(actions, ndims(actions)))
    #loop on samples, in principle this can be threaded
    for (l, (r, t, q, q_t, logπ, logμ, nsteps)) in enumerate(zip(map(tr -> eachslice(tr, dims = ndims(tr)), (rewards, terminals, qs, qs_t, logπs, logμs))..., k)) 
        δs = @views r[1:end-1] .+ p.γ .* (1 .- t[1:end-1]) .* q_t[2:end] .- q[1:end-1]
        cs = ci.(logπ, logμ)[1:(nsteps-1)]
        cs[1] = one(eltype(cs))
        cumprod!(cs,cs)
        target = sum(p.γ^(n-1) * cs[n] * δs[n] for n in 1:(nsteps-1))
        targets[l] = target
    end
    return targets
end

function q_targets(p::AbstractPolicy, traj::RetraceTrajectory, qnetwork::QNetwork, twinqnetwork::QNetwork, batch, k)
    states, actions, rewards, terminals, logμs = batch
    s = send_to_device(device(p), states) 
    a = send_to_device(device(p), actions)
    a_t = p(p.rng, s, is_sampling = true)
    q1s_t = qnetwork.target(s, a_t) |> send_to_host
    q2s_t = twinqnetwork.target(s, a_t) |> send_to_host
    qs_t = min.(q1s_t, q2s_t)
    qs = qnetwork(s, a) |> send_to_host
    logπs = p(s, a) |> send_to_host
    ci(logπ, logμ) = traj.λ*min(1, exp(logπ)/exp(logμ))
    targets = zeros(Float32, size(actions, ndims(actions)))
    #loop on samples, in principle this can be threaded
    for (l, (r, t, q, q_t, logπ, logμ, nsteps)) in enumerate(zip(map(tr -> eachslice(tr, dims = ndims(tr)), (rewards, terminals, qs, qs_t, logπs, logμs))..., k)) 
        δs = @views r[1:nsteps-1] .+ p.γ .* (1 .- t[1:nsteps-1]) .* q_t[2:nsteps] .- q[1:nsteps-1]
        cs = ci.(logπ, logμ)[1:(nsteps-1)]
        cs[1] = one(eltype(cs))
        cumprod!(cs,cs)
        target = sum(p.γ^(n-1) * cs[n] * δs[n] for n in 1:(nsteps-1))
        targets[l] = target
    end
    return targets
end