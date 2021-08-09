export MADDPGManager

"""
    MADDPGManager(; agents::Dict{<:Any, <:Agent}, args...)
Multi-agent Deep Deterministic Policy Gradient(MADDPG) implemented in Julia. Here only works for simultaneous games whose action space is discrete.
See the paper https://arxiv.org/abs/1706.02275 for more details.

# Keyword arguments
- `agents::Dict{<:Any, <:Agent{<:DDPGPolicy, <:AbstractTrajectory}}`, here each agent collects its own information. While updating the policy, each `critic` will assemble all agents' trajectory to update its own network.
- `batch_size::Int`
- `update_freq::Int`
- `update_step::Int`, count the step.
- `rng::AbstractRNG`.
"""
mutable struct MADDPGManager{P<:DDPGPolicy, T<:AbstractTrajectory} <: AbstractPolicy
    agents::Dict{<:Any, <:Agent{<:P, <:T}}
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::AbstractRNG
end

# for simultaneous game with a discrete action space.
function (π::MADDPGManager)(env::AbstractEnv)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end
    Dict((player, ceil(agent.policy(env))) for (player, agent) in π.agents)
end

function (π::MADDPGManager)(::PreEpisodeStage, ::AbstractEnv)
    for (_, agent) in π.agents
        if length(agent.trajectory) > 0
            pop!(agent.trajectory[:state])
            pop!(agent.trajectory[:action])
            if haskey(agent.trajectory, :legal_actions_mask)
                pop!(agent.trajectory[:legal_actions_mask])
            end
        end
    end
end

function (π::MADDPGManager)(::PreActStage, env::AbstractEnv, actions)
    # update each agent's trajectory
    for (player, agent) in π.agents
        push!(agent.trajectory[:state], state(env, player))
        push!(agent.trajectory[:action], actions[player])
        if haskey(agent.trajectory, :legal_actions_mask)
            lasm = legal_action_space_mask(env, player)
            push!(agent.trajectory[:legal_actions_mask], lasm)
        end
    end
    
    # update policy
    update!(π)
end

function (π::MADDPGManager)(::PostActStage, env::AbstractEnv)
    for (player, agent) in π.agents
        push!(agent.trajectory[:reward], reward(env, player))
        push!(agent.trajectory[:terminal], is_terminated(env))
    end
end

function (π::MADDPGManager)(::PostEpisodeStage, env::AbstractEnv)
    # collect state and dummy action to each agent's trajectory
    for (player, agent) in π.agents
        push!(agent.trajectory[:state], state(env, player))
        push!(agent.trajectory[:action], rand(action_space(env)))
        if haskey(agent.trajectory, :legal_actions_mask)
            lasm = legal_action_space_mask(env, player)
            push!(agent.trajectory[:legal_actions_mask], lasm)
        end
    end

    # update policy
    update!(π)
end

# update policy
function RLBase.update!(π::MADDPGManager)
    π.update_step += 1
    π.update_step % π.update_freq == 0 || return

    for (_, agent) in π.agents
        length(agent.trajectory) > agent.policy.update_after || return
        length(agent.trajectory) > π.batch_size || return
    end
    
    # get trainning data
    temp_player = rand(keys(π.agents))
    t = π.agents[temp_player].trajectory
    inds = rand(π.rng, 1:length(t), π.batch_size)
    batches = Dict((player, RLCore.fetch!(BatchSampler{SARTS}(π.batch_size), agent.trajectory, inds)) 
                for (player, agent) in π.agents)
    
    # get s, a, s′ for critic
    s = vcat((batches[player][1] for (player, _) in π.agents)...)
    a = vcat((batches[player][2] for (player, _) in π.agents)...)
    s′ = vcat((batches[player][5] for (player, _) in π.agents)...)

    # for training behavior_actor
    mu_actions = vcat(
        ((
            batches[player][1] |> # get personal state information
            x -> send_to_device(device(agent.policy.behavior_actor), x) |>
            agent.policy.behavior_actor |> send_to_host
        ) for (player, agent) in π.agents)...
    )
    # for training behavior_critic
    new_actions = vcat(
        ((
            batches[player][5] |> # batch[5] get new_state information
            x -> send_to_device(device(agent.policy.target_actor), x) |>
            agent.policy.target_actor |> send_to_host
        ) for (player, agent) in π.agents)...
    )

    for (player, agent) in π.agents
        p = agent.policy
        A = p.behavior_actor
        C = p.behavior_critic
        Aₜ = p.target_actor
        Cₜ = p.target_critic

        γ = p.γ
        ρ = p.ρ

        _device(x) = send_to_device(device(A), x)

        # Note that here default A, C, Aₜ, Cₜ on the same device.
        s, a, s′ = _device((s, a, s′))
        mu_actions = _device(mu_actions)
        new_actions = _device(new_actions)
        r = _device(batches[player][:reward])
        t = _device(batches[player][:terminal])

        qₜ = Cₜ(vcat(s′, new_actions)) |> vec
        y = r .+ γ .* (1 .- t) .* qₜ

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

        s, a, s′ = send_to_host((s, a, s′))
        mu_actions = send_to_host(mu_actions)
        new_actions = send_to_host(new_actions)
    end
end
