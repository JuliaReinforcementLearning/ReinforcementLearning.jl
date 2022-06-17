export TD3Policy, TD3Critic

struct TD3Critic
    critic_1::Flux.Chain
    critic_2::Flux.Chain
end
Flux.@functor TD3Critic
(c::TD3Critic)(s, a) = (inp = vcat(s, a); (c.critic_1(inp), c.critic_2(inp)))

mutable struct TD3Policy{
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
    update_freq::Int
    policy_freq::Int
    target_act_limit::Float64
    target_act_noise::Float64
    act_limit::Float64
    act_noise::Float64
    update_step::Int
    rng::R
    replay_counter::Int
    # for logging
    actor_loss::Float32
    critic_loss::Float32
end

"""
    TD3Policy(;kwargs...)

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
- `update_freq = 50`,
- `policy_freq = 2` # frequency in which the actor performs a gradient update_step and critic target is updated
- `target_act_limit = 1.0`, # noise added to actor target
- `target_act_noise = 0.1`, # noise added to actor target
- `act_limit = 1.0`, # noise added when outputing action
- `act_noise = 0.1`, # noise added when outputing action
- `update_step = 0`,
- `rng = Random.GLOBAL_RNG`,
"""
function TD3Policy(;
    behavior_actor,
    behavior_critic,
    target_actor,
    target_critic,
    start_policy,
    γ = 0.99f0,
    ρ = 0.995f0,
    batch_size = 64,
    start_steps = 10000,
    update_after = 1000,
    update_freq = 50,
    policy_freq = 2,
    target_act_limit = 1.0,
    target_act_noise = 0.1,
    act_limit = 1.0,
    act_noise = 0.1,
    update_step = 0,
    rng = Random.GLOBAL_RNG,
)
    copyto!(behavior_actor, target_actor)  # force sync
    copyto!(behavior_critic, target_critic)  # force sync
    TD3Policy(
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
        update_freq,
        policy_freq,
        target_act_limit,
        target_act_noise,
        act_limit,
        act_noise,
        update_step,
        rng,
        1, # keep track of numbers of replay
        0.0f0,
        0.0f0,
    )
end

# TODO: handle Training/Testing mode
function (p::TD3Policy)(env)
    p.update_step += 1

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.behavior_actor)
        s = state(env)
        s = Flux.unsqueeze(s, ndims(s) + 1)
        action = p.behavior_actor(send_to_device(D, s)) |> vec |> send_to_host
        clamp(action[] + randn(p.rng) * p.act_noise, -p.act_limit, p.act_limit)
    end
end

function RLBase.update!(
    p::TD3Policy,
    traj::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    length(traj) > p.update_after || return
    p.update_step % p.update_freq == 0 || return
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
    update!(p, batch)
end

function RLBase.update!(p::TD3Policy, batch::NamedTuple{SARTS})
    to_device(x) = send_to_device(device(p.behavior_actor), x)
    s, a, r, t, s′ = to_device(batch)

    actor = p.behavior_actor
    critic = p.behavior_critic

    # !!! we have several assumptions here, need revisit when we have more complex environments
    # state is vector
    # action is scalar
    target_noise =
        clamp.(
            randn(p.rng, Float32, 1, p.batch_size) .* p.target_act_noise,
            -p.target_act_limit,
            p.target_act_limit,
        ) |> to_device
    # add noise and clip to act_limit bounds
    a′ = clamp.(p.target_actor(s′) + target_noise, -p.act_limit, p.act_limit)

    q_1′, q_2′ = p.target_critic(s′, a′)
    y = r .+ p.γ .* (1 .- t) .* (min.(q_1′, q_2′) |> vec)

    # ad-hoc fix to https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues/624
    if ndims(a) == 1
        a = Flux.unsqueeze(a, 1)
    end

    gs1 = gradient(Flux.params(critic)) do
        q1, q2 = critic(s, a)
        loss = mse(q1 |> vec, y) + mse(q2 |> vec, y)
        ignore() do
            p.critic_loss = loss
        end
        loss
    end
    update!(critic, gs1)

    if p.replay_counter % p.policy_freq == 0
        gs2 = gradient(Flux.params(actor)) do
            actions = actor(s)
            loss = -mean(critic.model.critic_1(vcat(s, actions)))
            ignore() do
                p.actor_loss = loss
            end
            loss
        end
        update!(actor, gs2)
        # polyak averaging
        for (dest, src) in zip(
            Flux.params([p.target_actor, p.target_critic]),
            Flux.params([actor, critic]),
        )
            dest .= p.ρ .* dest .+ (1 - p.ρ) .* src
        end
        p.replay_counter = 1
    end
    p.replay_counter += 1
end
