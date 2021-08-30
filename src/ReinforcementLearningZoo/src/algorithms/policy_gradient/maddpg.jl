export MADDPGManager

"""
    MADDPGManager(; agents::Dict{<:Any, <:Agent}, args...)
Multi-agent Deep Deterministic Policy Gradient(MADDPG) implemented in Julia. By default, `MADDPGManager` uses for simultaneous environments with [continuous action space](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies).
See the paper https://arxiv.org/abs/1706.02275 for more details.

# Keyword arguments
- `agents::Dict{<:Any, <:Agent}`, here each agent collects its own information. While updating the policy, each **critic** will assemble all agents' 
  trajectory to update its own network. **Note that** here the policy of the `Agent` should be `DDPGPolicy` wrapped by `NamedPolicy`, see the relative 
  experiment([`MADDPG_KuhnPoker`](https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker/#JuliaRL\\_MADDPG\\_KuhnPoker) or [`MADDPG_SpeakerListener`](https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_SpeakerListener/#JuliaRL\\_MADDPG\\_SpeakerListener)) for references.
- `traces`, set to `SARTS` if you are apply to an environment of `MINIMAL_ACTION_SET`, or `SLARTSL` if you are to apply to an environment of `FULL_ACTION_SET`.
- `batch_size::Int`
- `update_freq::Int`
- `update_step::Int`, count the step.
- `rng::AbstractRNG`.
"""
mutable struct MADDPGManager <: AbstractPolicy
    agents::Dict{<:Any, <:Agent}
    traces
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::AbstractRNG
end

# used for simultaneous environments.
function (π::MADDPGManager)(env::AbstractEnv)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end
    Dict(
        player => agent.policy(env)
    for (player, agent) in π.agents)
end

function (π::MADDPGManager)(stage::Union{PreEpisodeStage, PostActStage}, env::AbstractEnv)
    # only need to update trajectory.
    for (_, agent) in π.agents
        update!(agent.trajectory, agent.policy, env, stage)
    end
end

function (π::MADDPGManager)(stage::PreActStage, env::AbstractEnv, actions)
    # update each agent's trajectory.
    for (player, agent) in π.agents
        update!(agent.trajectory, agent.policy, env, stage, actions[player])
    end
    
    # update policy
    update!(π, env)
end

function (π::MADDPGManager)(stage::PostEpisodeStage, env::AbstractEnv)
    # collect state and a dummy action to each agent's trajectory here.
    for (_, agent) in π.agents
        update!(agent.trajectory, agent.policy, env, stage)
    end

    # update policy
    update!(π, env)
end

# update policy
function RLBase.update!(π::MADDPGManager, env::AbstractEnv)
    π.update_step += 1
    π.update_step % π.update_freq == 0 || return

    for (_, agent) in π.agents
        length(agent.trajectory) > agent.policy.policy.update_after || return
        length(agent.trajectory) > π.batch_size || return
    end
    
    # get training data
    temp_player = collect(keys(π.agents))[1]
    t = π.agents[temp_player].trajectory
    inds = rand(π.rng, 1:length(t), π.batch_size)
    batches = Dict((player, RLCore.fetch!(BatchSampler{π.traces}(π.batch_size), agent.trajectory, inds)) 
                for (player, agent) in π.agents)
    
    # get s, a, s′ for critic
    s = vcat((batches[player][:state] for (player, _) in π.agents)...)
    a = vcat((batches[player][:action] for (player, _) in π.agents)...)
    s′ = vcat((batches[player][:next_state] for (player, _) in π.agents)...)

    for (player, agent) in π.agents
        p = agent.policy.policy # get agent's concrete DDPGPolicy.

        A = p.behavior_actor
        C = p.behavior_critic
        Aₜ = p.target_actor
        Cₜ = p.target_critic

        γ = p.γ
        ρ = p.ρ

        # by default A, C, Aₜ, Cₜ on the same device.
        batches, s, a, s′ = send_to_device(device(A), (batches, s, a, s′))
        r = batches[player][:reward]
        t = batches[player][:terminal]
        # for training behavior_actor.
        mu_actions = vcat(
            ((
                batches[p][:next_state] |>
                a.policy.policy.behavior_actor
            ) for (p, a) in π.agents)...
        )
        # for training behavior_critic.
        new_actions = vcat(
            ((
                batches[p][:next_state] |> 
                a.policy.policy.target_actor
            ) for (p, a) in π.agents)...
        )

        if π.traces == SLARTSL
            # Note that by default **MADDPG** is used for the environments with continuous action space, and `legal_action_space_mask` is 
            # defined in the environments with discrete action space. So we need `env.action_mapping` to transform the actions 
            # getting from the trajectory.
            @assert env isa ActionTransformedEnv

            mask = batches[player][:next_legal_actions_mask]
            mu_l′ = Flux.batch(
                (begin
                    actions = env.action_mapping(mu_actions[:, i])
                    mask[actions[player]]
                end for i = 1:π.batch_size)
            )
            new_l′ = Flux.batch(
                (begin
                    actions = env.action_mapping(new_actions[:, i])
                    mask[actions[player]]
                end for i = 1:π.batch_size)
            )
        end

        qₜ = Cₜ(vcat(s′, new_actions)) |> vec
        if π.traces == SLARTSL
            qₜ .+= ifelse.(new_l′, 0.0f0, typemin(Float32))
        end
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
            v = C(vcat(s, mu_actions)) |> vec
            if π.traces == SLARTSL
                v .+= ifelse.(mu_l′, 0.0f0, typemin(Float32))
            end
            reg = mean(A(batches[player][:state]) .^ 2)
            loss = -mean(v) +  reg * 1e-3
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
