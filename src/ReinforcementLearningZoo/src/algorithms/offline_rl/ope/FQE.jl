export FQE

mutable struct FQE{
    P<:GaussianNetwork,
    C<:NeuralNetworkApproximator,
    C_T<:NeuralNetworkApproximator,
    R<:AbstractRNG,
 } <: AbstractLearner
    policy::P
    q_network::C
    target_q_network::C_T
    n_evals::Int
    γ::Float32
    batch_size::Int
    update_freq::Int
    update_step::Int
    tar_update_freq::Int
    rng::R
    #logging
    loss::Float32
end

"""
    FQE(;kwargs...)

See [Hyperparameter Selection for Offline Reinforcement Learning](https://arxiv.org/abs/2007.09055)

# Keyword arguments
- `policy`, the policy for which FQE should be performed.
- `q_network`, critic to evaluate the Q value of `state`, `action` pair.
- `target_q_network`, target critic used for evaluating target Q values.
- `n_evals::Int`, number of evaluations to perform to return the performance of the policy.
- `γ::Float32 = 0.99f0`, discount factor.
- `batch_size::Int = 32`.
- `update_freq::Int = 50`, frequency of updating the `target_q_network`.
- `update_step::Int = 0`.
- `tar_update_freq::Int = 50`
- `rng::AbstractRNG = Range.GLOBAL_RNG`.

`policy` is expected to be a pre-trained [`GaussianNetwork`](@ref) with a particular choice of hyperparameters
preferrably trained using the same `dataset`.
"""
function FQE(;
    policy,
    q_network,
    target_q_network,
    n_evals=20,
    γ=0.99f0,
    batch_size=32,
    update_freq=1,
    update_step=0,
    tar_update_freq = 50,
    rng=Random.GLOBAL_RNG,
)
    copyto!(q_network, target_q_network) #force sync
    FQE(
        policy,
        q_network,
        target_q_network,
        n_evals,
        γ,
        batch_size,
        update_freq,
        update_step,
        tar_update_freq,
        rng,
        0.0f0,
    )
end

Flux.functor(x::FQE) = (Q = x.q_network, Qₜ = x.target_q_network),
y -> begin
    x = @set x.q_network = y.Q
    x = @set x.target_q_network = y.Qₜ
    x
end

function (l::FQE)(env)
    s = send_to_device(device(l.policy), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    action = dropdims(l.policy(l.rng, s; is_sampling=true), dims=2)
end

function (l::FQE)(env, ::Val{:Eval})
    results = []
    D = device(l.policy)
    for _ in 1:l.n_evals
        reset!(env)
        s = send_to_device(D, state(env))
        s = Flux.unsqueeze(s, ndims(s)+1)
        a = dropdims(l.policy(l.rng, s; is_sampling=true), dims=2)
        input = vcat(s, a)
        result = l.q_network(input)
        push!(results, result[])
    end
    mean(results), results
end

function (l::FQE)(state::AbstractArray, action::AbstractArray)
    D = device(l.q_network)
    s = send_to_device(D, state)
    a = send_to_device(D, reshape(action, :, l.batch_size))
    input = vcat(s, a)
    value = l.q_network(input)
end

function RLBase.update!(l::FQE, batch::NamedTuple{SARTS})
    policy = l.policy
    Q, Qₜ = l.q_network, l.target_q_network
    
    D = device(Q)
    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    γ = l.γ
    batch_size = l.batch_size

    loss_func = Flux.Losses.mse

    q′ = Qₜ(vcat(s′, policy(s′)[1])) |> vec
    
    target = r .+ γ .* (1 .- t) .* q′

    gs = gradient(params(Q)) do
        q = Q(vcat(s, reshape(a, :, batch_size))) |> vec
        loss = loss_func(q, target)
        Zygote.ignore() do
            l.loss = loss
        end
        loss
    end
    Flux.Optimise.update!(Q.optimizer, params(Q), gs)

    if l.update_step % l.tar_update_freq == 0
        Qₜ = deepcopy(Q)
    end
end