export DDPGPolicy
using Functors

mutable struct DDPGPolicy{
    A<:Approximator{<:TwinNetwork},
    C<:Approximator{<:TwinNetwork},
    P,
    R<:AbstractRNG,
} <: AbstractPolicy
    actor::A
    critic::C
    γ::Float32
    na::Int
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_freq::Int
    act_upper_limit::Float64
    act_lower_limit::Float64
    act_noise::Float64
    update_step::Int
    rng::R
    training::Bool
    # for logging
    actor_loss::Float32
    critic_loss::Float32
end

Functors.functor(x::DDPGPolicy) = (
    a=x.actor,
    c=x.critic
),
y -> begin
    x.actor = y.a
    x.critic = y.c
    x
end

"""
    DDPGPolicy(;kwargs...)

# Keyword arguments

- `actor`,
- `critic`,
- `start_policy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_freq = 50`,
- `act_upper_limit = 1.0`,
- `act_lower_limit = 0.0`,
- `act_noise = 0.1`,
- `update_step = 0`,
- `rng = Random.GLOBAL_RNG`,
"""
function DDPGPolicy(;
    actor,
    critic,
    start_policy,
    γ=0.99f0,
    na=1,
    batch_size=32,
    start_steps=10000,
    update_after=1000,
    update_freq=50,
    act_upper_limit=1.0,
    act_lower_limit=0.0,
    act_noise=0.1,
    update_step=0,
    rng=Random.GLOBAL_RNG,
    training=true
)
    DDPGPolicy(
        actor,
        critic,
        γ,
        na,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_freq,
        act_upper_limit,
        act_lower_limit,
        act_noise,
        update_step,
        rng,
        training,
        0.0f0,
        0.0f0
    )
end

function (p::DDPGPolicy)(env::AbstractEnv)
    if p.training 
        p.update_step += 1
    end

    D = device(p.actor)
    s = DynamicStyle(env) == SEQUENTIAL ? state(env) : state(env, player)
    s = Flux.unsqueeze(s, dims=ndims(s) + 1)
    actions = p.actor(send_to_device(D, s)) |> vec |> send_to_host
    c =
        clamp.(
            actions .+ randn(p.rng, p.na) .* repeat([p.act_noise], p.na),
            p.act_lower_limit,
            p.act_upper_limit,
        )
    c = p.na == 1 ? c[1] : c
    if !p.training
        c
        # @info c
    elseif p.update_after >= p.update_step
        a = p.start_policy(env)
        # @info a
    else
        # @info c
        c
    end
end


function RLBase.optimise!(p::DDPGPolicy, batch::NamedTuple{SS′ART})
    if !p.training
        return
    end
    s, s_next, a, r, t = send_to_device(device(p), batch)

    AA = p.actor
    A = AA.model.source
    Aₜ = AA.model.target
    AC = p.critic
    C = AC.model.source
    Cₜ = AC.model.target

    γ = p.γ


    a_next = Aₜ(s_next)
    qₜ = Cₜ(vcat(s_next, a_next)) |> vec
    y = r .+ γ .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, dims=ndims(a) + 1)

    gs1 = gradient(params(AC)) do
        q = C(vcat(s, a)) |> vec
        loss = mean((y .- q) .^ 2)
        ignore_derivatives() do
            p.critic_loss = loss
            # @info loss
        end
        loss
    end

    optimise!(AC, gs1)

    gs2 = gradient(params(AA)) do
        loss = -mean(C(vcat(s, A(s))))
        ignore_derivatives() do
            p.actor_loss = loss
            # @info loss
        end
        loss
    end

    optimise!(AA, gs2)
end
