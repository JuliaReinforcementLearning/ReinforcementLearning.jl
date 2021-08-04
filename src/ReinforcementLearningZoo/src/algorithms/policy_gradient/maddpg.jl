export MADDPGManager

"""
    MADDPGManager(; agents::Dict{<:Any, <:DDPGPolicy}, args...)

Multi-agent Deep Deterministic Policy Gradient implemented in Julia. 
See the paper https://arxiv.org/abs/1706.02275 for more details.

# Keyword arguments

- `agents::Dict{<:Any, <:Agent{<:DDPGPolicy, <:AbstractTrajectory}}`, here the `trajectory` only collect the agent's personal information state.
- `batch_size::Int`
- `update_after::Int`
- `update_freq::Int`
- `step_counter::Int`, count the step.
- `rng::AbstractRNG`.
"""
mutable struct MADDPGManager{P<:DDPGPolicy, T<:AbstractTrajectory} <: AbstractPolicy
    agents::Dict{<:Any, <:Agent{<:P, <:T}}
    batch_size::Int
    update_after::Int
    update_freq::Int
    step_counter::Int
    rng::AbstractRNG
end

# for discrete action
function (p::MADDPGManager)(env::AbstractEnv)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end
    [ceil(agent.policy(env)) for (_, agent) in p.agents]
end

# update trajectory
function RLBase.update!(
    traj::AbstractTrajectory,
    policy::MADDPGManager,
    env::AbstractEnv,
    ::PreActStage,
    actions::Vector
)
    # update global trajectory (for training critic)
    push!(traj[:state], state(env, chance_player(env)))
    push!(traj[:action], actions)
    if haskey(traj, :legal_actions_mask)
        push!(traj[:legal_actions_mask], legal_action_space_mask(env))
    end

    # update personal trajectory (only need to get information state)
    for (player, agent) in policy.agents
        actor_state = state(env, player)
        push!(agent.trajectory[:state], actor_state)
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    ::MADDPGManager,
    env::AbstractEnv,
    ::PostActStage,
)
    push!(trajectory[:reward], reward(env, chance_player(env)))
    push!(trajectory[:terminal], is_terminated(env))
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::MADDPGManager,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    if length(trajectory) > 0
        pop!(trajectory[:state])
        pop!(trajectory[:action])
        if haskey(trajectory, :legal_actions_mask)
            pop!(trajectory[:legal_actions_mask])
        end
    end

    for (_, agent) in policy.agents
        if length(agent.trajectory) > 0
            pop!(agent.trajectory[:state])
        end
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::MADDPGManager,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    actions = rand(action_space(env), length(policy.agents))

    push!(trajectory[:state], state(env, chance_player(env)))
    push!(trajectory[:action], actions)
    if haskey(trajectory, :legal_actions_mask)
        push!(trajectory[:legal_actions_mask], legal_action_space_mask(env))
    end

    for (player, agent) in policy.agents
        actor_state = state(env, player)
        push!(agent.trajectory[:state], actor_state)
    end    
end

# update policy
function RLBase.update!(
    policy::MADDPGManager, 
    traj::AbstractTrajectory,
    ::AbstractEnv, 
    ::PreActStage,
)
    length(traj) > policy.update_after || return
    policy.step_counter += 1
    policy.step_counter % policy.update_freq == 0 || return
    
    inds, batch = sample(policy.rng, traj, BatchSampler{SARTS}(policy.batch_size))

    # for training behavior_actor
    mu_actions = vcat(
        ((
            consecutive_view(agent.trajectory[:state], inds) |> 
            x -> send_to_device(device(agent.policy.behavior_actor), x) |>
            agent.policy.behavior_actor |> send_to_host
        ) for (_, agent) in policy.agents)...
    )
    # for training behavior_critic
    new_actions = vcat(
        ((
            consecutive_view(agent.trajectory[:state], (inds .+ 1)) |> 
            x -> send_to_device(device(agent.policy.target_actor), x) |>
            agent.policy.target_actor |> send_to_host
        ) for (_, agent) in policy.agents)...
    )

    for (player, agent) in policy.agents
        p = agent.policy

        A = p.behavior_actor
        C = p.behavior_critic
        Aₜ = p.target_actor
        Cₜ = p.target_critic

        γ = p.γ
        ρ = p.ρ

        _device(x) = send_to_device(device(A), x)

        # default A, C, Aₜ, Cₜ on the same device
        s, a, r, t, s′ = _device(batch)
        mu_actions = _device(mu_actions)
        new_actions = _device(new_actions)

        qₜ = Cₜ(vcat(s′, new_actions)) |> vec
        y = r[player, :] .+ γ .* (1 .- t) .* qₜ

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

        batch = send_to_host(batch)
        mu_actions = send_to_host(mu_actions)
        new_actions = send_to_host(new_actions)
    end
end
