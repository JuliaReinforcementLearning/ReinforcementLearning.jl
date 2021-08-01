export MADDPGPolicy

mutable struct MADDPGPolicy <: AbstractPolicy
    agents::Dict{<:Any, <:DDPGPolicy}
    batch_size::Int
    update_after::Int
    update_freq::Int
    step::Int
    rng::AbstractRNG
end

function (p::MADDPGPolicy)(env::AbstractEnv)
    p.step += 1
    return [ceil(policy(env)) for policy in values(p.agents)]
end

function update!(
    policy::MADDPGPolicy, 
    traj::AbstractTrajectory,
    ::AbstractEnv, 
    ::PreActStage,
    )
    length(traj) > policy.update_after || return
    policy.step % policy.update_every == 0 || return

    _, batch = sample(policy.rng, traj, BatchSampler{SARTS}(policy.batch_size))
    _device = device(policy.agents[1])
    s, a, r, t, s′ = send_to_device(_device, batch)

    mu_actions = vcat(p.behavior_actor(s) for p in values(policy.agents))
    new_actions = vcat(p.target_actor(s′) for p in values(policy.agents))

    for p in values(policy.agents)
        A = p.behavior_actor
        C = p.behavior_critic
        Aₜ = p.target_actor
        Cₜ = p.target_critic

        γ = p.γ
        ρ = p.ρ

        qₜ = Cₜ(vcat(s′, new_actions)) |> vec
        y = r .+ γ .* (1 .- t) .* qₜ # where r should be with respect ot specific agent

        gs1 = gradient(Flux.params(C)) do
            q = C(vcat(s, a)) |> vec
            loss = mean((y .- q) .^ 2)
            ignore() do
                p.critic_loss = loss
            end
            loss
        end

        update!(C, gs1)

        gs2 = gradient(Flux.params(A)) do
            loss = -mean(C(vcat(s, mu_actions)))
            ignore() do
                p.actor_loss = loss
            end
            loss
        end

        update!(A, gs2)

        # polyak averaging
        for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
            dest .= ρ .* dest .+ (1 - ρ) .* src
        end
    end
end

# run function
function Base.run(
    policy::Agent{<:MADDPGPolicy, <:AbstractTrajectory},
    env::AbstractEnv,
    stop_condition = StopAfterEpisode(1),
    hook = EmptyHook(),
)

    is_stop = false

    while !is_stop
        RLBase.reset!(env)
        hook(PRE_EPISODE_STAGE, policy, env)

        while !is_terminated(env) # one episode
            actions = policy(env)
           
            policy(PRE_ACT_STAGE, env, actions)
            hook(PRE_ACT_STAGE, policy, env, actions)

            for action in actions
                env(action)
            end
            
            policy(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, policy, env)

            if stop_condition(policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(POST_EPISODE_STAGE, env)
            hook(POST_EPISODE_STAGE, policy, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, policy, env)
    hook
end
