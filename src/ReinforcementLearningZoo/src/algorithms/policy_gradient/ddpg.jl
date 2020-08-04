export DDPGPolicy

using Random
using Flux

mutable struct DDPGPolicy{
    BA<:NeuralNetworkApproximator,
    BC<:NeuralNetworkApproximator,
    TA<:NeuralNetworkApproximator,
    TC<:NeuralNetworkApproximator,
    P,
    R<:AbstractRNG,
} <: AbstractPolicy

    behavior_actor::BA
    behavior_critic::BC
    target_actor::TA
    target_critic::TC
    γ::Float32
    ρ::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_every::Int
    act_limit::Float64
    act_noise::Float64
    step::Int
    rng::R
end

"""
    DDPGPolicy(;kwargs...)

# Keyword arguments

- `behavior_actor`,
- `behavior_critic`,
- `target_actor`,
- `target_critic`,
- `start_policy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_every = 50`,
- `act_limit = 1.0`,
- `act_noise = 0.1`,
- `step = 0`,
- `rng = Random.GLOBAL_RNG`,
"""
function DDPGPolicy(;
    behavior_actor,
    behavior_critic,
    target_actor,
    target_critic,
    start_policy,
    γ = 0.99f0,
    ρ = 0.995f0,
    batch_size = 32,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    act_limit = 1.0,
    act_noise = 0.1,
    step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(behavior_actor, target_actor)  # force sync
    copyto!(behavior_critic, target_critic)  # force sync
    DDPGPolicy(
        behavior_actor,
        behavior_critic,
        target_actor,
        target_critic,
        γ,
        ρ,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_every,
        act_limit,
        act_noise,
        step,
        rng,
    )
end

# TODO: handle Training/Testing mode
function (p::DDPGPolicy)(env)
    p.step += 1

    if p.step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.behavior_actor)
        s = get_state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        action = p.behavior_actor(send_to_device(D, s)) |> vec |> send_to_host
        clamp(action[] + randn(p.rng) * p.act_noise, -p.act_limit, p.act_limit)
    end
end

function RLBase.update!(p::DDPGPolicy, t::CircularCompactSARTSATrajectory)
    length(t) > p.update_after || return
    p.step % p.update_every == 0 || return

    inds = rand(p.rng, 1:(length(t)-1), p.batch_size)
    SARTS = (:state, :action, :reward, :terminal, :next_state)
    s, a, r, t, s′ = map(x -> select_last_dim(get_trace(t, x), inds), SARTS)

    A = p.behavior_actor
    C = p.behavior_critic
    Aₜ = p.target_actor
    Cₜ = p.target_critic

    γ = p.γ
    ρ = p.ρ


    # !!! we have several assumptions here, need revisit when we have more complex environments
    # state is vector
    # action is scalar
    a′ = Aₜ(s′)
    qₜ = Cₜ(vcat(s′, a′)) |> vec
    y = r .+ γ .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, 1)

    gs1 = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        mean((y .- q) .^ 2)
    end

    update!(C, gs1)

    gs2 = gradient(Flux.params(A)) do
        -mean(C(vcat(s, A(s))))
    end

    update!(A, gs2)

    # polyak averaging
    for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end
