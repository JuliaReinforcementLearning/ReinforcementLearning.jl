export DeepCFR


"""
    DeepCFR(;kwargs...)

Symbols used here follow the paper: [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164)

# Keyword arguments

- `K`, number of traversal.
- `t`, number of iteration.
- `Π`, the policy network.
- `V`, a dictionary of each player's advantage network.
- `MΠ`, a strategy memory.
- `MV`, a dictionary of each player's advantage memory.
- `reinitialize_freq=1`, the frequency of re-initializing the value networks.
"""
Base.@kwdef mutable struct DeepCFR{TP,TV,TMP,TMV,I,R,P} <: AbstractCFRPolicy
    Π::TP
    V::TV
    MΠ::TMP
    MV::TMV
    K::Int = 20
    t::Int = 1
    reinitialize_freq::Int = 1
    batch_size_V::Int = 32
    batch_size_Π::Int = 32
    n_training_steps_V::Int = 1
    n_training_steps_Π::Int = 1
    rng::R = Random.GLOBAL_RNG
    initializer::I = glorot_normal(rng)
    max_grad_norm::Float32 = 10.0f0
    # for logging
    Π_losses::Vector{Float32} = zeros(Float32, n_training_steps_Π)
    V_losses::Dict{P,Vector{Float32}} =
        Dict(k => zeros(Float32, n_training_steps_V) for (k, _) in MV)
    Π_norms::Vector{Float32} = zeros(Float32, n_training_steps_Π)
    V_norms::Dict{P,Vector{Float32}} =
        Dict(k => zeros(Float32, n_training_steps_V) for (k, _) in MV)
end

function RLBase.prob(π::DeepCFR, env::AbstractEnv)
    I = send_to_device(device(π.Π), state(env))
    m = send_to_device(device(π.Π), ifelse.(legal_action_space_mask(env), 0.0f0, -Inf32))
    logits = π.Π(Flux.unsqueeze(I, ndims(I) + 1)) |> vec
    σ = softmax(logits .+ m)
    send_to_host(σ)
end

(π::DeepCFR)(env::AbstractEnv) =
    sample(π.rng, action_space(env), Weights(prob(π, env), 1.0))

"Run one interation"
function RLBase.update!(π::DeepCFR, env::AbstractEnv)
    for p in players(env)
        if p != chance_player(env)
            for k in 1:π.K
                external_sampling!(π, copy(env), p)
            end
            update_advantage_networks(π, p)
        end
    end
    π.t += 1
end

"Update Π (policy network)"
function RLBase.update!(π::DeepCFR)
    Π = π.Π
    Π_losses = π.Π_losses
    Π_norms = π.Π_norms
    D = device(Π)
    MΠ = π.MΠ
    ps = Flux.params(Π)

    for x in ps
        x .= π.initializer(size(x)...)
    end

    for i in 1:π.n_training_steps_Π
        batch_inds = rand(π.rng, 1:length(MΠ), π.batch_size_Π)
        I = send_to_device(D, Flux.batch([MΠ[:I][i] for i in batch_inds]))
        σ = send_to_device(D, Flux.batch([MΠ[:σ][i] for i in batch_inds]))
        t = send_to_device(D, Flux.batch([MΠ[:t][i] / π.t for i in batch_inds]))
        m = send_to_device(
            D,
            Flux.batch([ifelse.(MΠ[:m][i], 0.0f0, -Inf32) for i in batch_inds]),
        )
        gs = gradient(ps) do
            logits = Π(I) .+ m
            loss = mean(reshape(t, 1, :) .* ((σ .- softmax(logits)) .^ 2))
            ignore() do
                # println(σ, "!!!",m, "===", Π(I))
                Π_losses[i] = loss
            end
            loss
        end
        Π_norms[i] = clip_by_global_norm!(gs, ps, π.max_grad_norm)
        update!(Π, gs)
    end
end

"Update advantage network"
function update_advantage_networks(π, p)
    V = π.V[p]
    V_losses = π.V_losses[p]
    V_norms = π.V_norms[p]
    MV = π.MV[p]
    if π.t % π.reinitialize_freq == 0
        for x in Flux.params(V)
            # TODO: inplace
            x .= π.initializer(size(x)...)
        end
    end
    if length(MV) >= π.batch_size_V
        for i in 1:π.n_training_steps_V
            batch_inds = rand(π.rng, 1:length(MV), π.batch_size_V)
            I = send_to_device(device(V), Flux.batch([MV[:I][i] for i in batch_inds]))
            r̃ = send_to_device(device(V), Flux.batch([MV[:r̃][i] for i in batch_inds]))
            t = send_to_device(device(V), Flux.batch([MV[:t][i] / π.t for i in batch_inds]))
            m = send_to_device(device(V), Flux.batch([MV[:m][i] for i in batch_inds]))
            ps = Flux.params(V)
            gs = gradient(ps) do
                loss = mean(reshape(t, 1, :) .* ((r̃ .- V(I) .* m) .^ 2))
                ignore() do
                    V_losses[i] = loss
                end
                loss
            end
            V_norms[i] = clip_by_global_norm!(gs, ps, π.max_grad_norm)
            update!(V, gs)
        end
    end
end

"CFR Traversal with External Sampling"
function external_sampling!(π::DeepCFR, env::AbstractEnv, p)
    if is_terminated(env)
        reward(env, p)
    elseif current_player(env) == chance_player(env)
        env(rand(π.rng, action_space(env)))
        external_sampling!(π, env, p)
    elseif current_player(env) == p
        V = π.V[p]
        s = state(env)
        I = send_to_device(device(V), Flux.unsqueeze(s, ndims(s) + 1))
        A = action_space(env)
        m = legal_action_space_mask(env)
        σ = masked_regret_matching(V(I) |> send_to_host |> vec, m)
        v = zeros(length(σ))
        v̄ = 0.0
        for i in 1:length(m)
            if m[i]
                v[i] = external_sampling!(π, child(env, A[i]), p)
                v̄ += σ[i] * v[i]
            end
        end
        push!(π.MV[p], I = s, t = π.t, r̃ = (v .- v̄) .* m, m = m)
        v̄
    else
        V = π.V[current_player(env)]
        s = state(env)
        I = send_to_device(device(V), Flux.unsqueeze(s, ndims(s) + 1))
        A = action_space(env)
        m = legal_action_space_mask(env)
        σ = masked_regret_matching(V(I) |> send_to_host |> vec, m)
        push!(π.MΠ, I = s, t = π.t, σ = σ, m = m)
        a = sample(π.rng, A, Weights(σ, 1.0))
        env(a)
        external_sampling!(π, env, p)
    end
end

"This is the specific regret matching method used in DeepCFR"
function masked_regret_matching(v, m)
    v⁺ = max.(v .* m, 0.0f0)
    s = sum(v⁺)
    if s > 0
        v⁺ ./= s
    else
        fill!(v⁺, 0.0f0)
        v⁺[findmax(v, m)[2]] = 1.0
    end
    v⁺
end
