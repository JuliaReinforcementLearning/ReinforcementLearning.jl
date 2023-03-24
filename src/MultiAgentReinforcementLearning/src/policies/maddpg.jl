export MADDPGManager
using BSON
using Flux
using Statistics

"""
    MADDPGManager(; agents::Dict{<:Any, <:Agent}, args...)
Multi-agent Deep Deterministic Policy Gradient(MADDPG) implemented in Julia. By default, `MADDPGManager` uses for simultaneous environments with [continuous action space](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies).
See the paper https://arxiv.org/abs/1706.02275 for more details.

# Keyword arguments
- `agents::Dict{S, <: AbstractPolicy}`, here each agent collects its own information. While updating the policy, each **critic** will assemble all agents' 
  trajectory to update its own network. **Note that** here the policy of the `Agent` should be `DDPGPolicy` wrapped by `NamedPolicy`, see the relative 
  experiment([`MADDPG_KuhnPoker`](https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_KuhnPoker/#JuliaRL\\_MADDPG\\_KuhnPoker) or [`MADDPG_SpeakerListener`](https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_SpeakerListener/#JuliaRL\\_MADDPG\\_SpeakerListener)) for references.
- `update_freq::Int`
- `update_step::Int`, count the step.
"""
mutable struct MADDPGManager{T <: AbstractPolicy, S} <: AbstractPolicy
    agents::Dict{S, T}
    update_freq::Int
    update_step::Int
end

# used for simultaneous environments.
function (π::MADDPGManager)(env::AbstractEnv)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end
    Dict(
        player => agent(env)
        for (player, agent) in π.agents)
end

function (π::MADDPGManager{<: Agent, <:Any})(::PreActStage, env::AbstractEnv)
    for (player, agent) in π.agents
        update!(agent, state(env, player))
    end
    
end

function (π::MADDPGManager{<: Agent, <: Any})(::PostActStage, env::AbstractEnv)
    for (player, agent) in π.agents
        update!(agent.cache, reward(env, player), is_terminated(env))
    end
end

function (π::MADDPGManager)(stage::AbstractStage, env::AbstractEnv)
    for (_, agent) in π.agents
        agent(stage, env)
    end
end



# # update policy
function RLBase.optimise!(π::MADDPGManager)
    π.update_step % π.update_freq == 0 || return
    π.update_step += 1

    for (_, agent) in π.agents
        length(agent.trajectory.container) > agent.policy.update_after || return
    end

    # get training data
    batches = Dict((player, [b for b in agent.trajectory])
                   for (player, agent) in π.agents)

    # do not optimize if one player has empty batch
    all(length(s[2]) != 0 for s in batches) || return
    # make sure that all agents have the same amount of batches
    @assert all(length(s[2]) == length(b[2]) 
    for (s,b) in Base.Iterators.product(batches, batches) 
    if s[1] != b[1])
    
    # n_batches = length(first(batches)[2]) since all agents
    # have the same amount of batches. Otherwise, this algorithm
    # will not be able to be trained accordingly unless dummy actions
    # are chosen based on the current state of the other agent`s current policy
    # state
    for i ∈ 1:length(first(batches)[2])
      optimise!(π, Dict(player => batches[player][i] for (player, ) in π.agents))
    end
    
end

function RLBase.optimise!(π::MADDPGManager, player_batches)
    # get s, a, s′ for critic
    s = vcat((player_batches[player][:state] for (player, _) in π.agents)...)
    a = vcat((player_batches[player][:action] for (player, _) in π.agents)...)
    s′ = vcat((player_batches[player][:next_state] for (player, _) in π.agents)...)

    for (player, agent) in π.agents
        p = agent.policy # get agent's concrete DDPGPolicy.

        AA = p.actor
        A = AA.model.source
        Aₜ = AA.model.target
        AC = p.critic
        C = AC.model.source
        Cₜ = AC.model.target

        γ = p.γ

        # by default A, C, Aₜ, Cₜ on the same device.
        player_batches, s, a, s′ = send_to_device(device(AA), (player_batches, s, a, s′))
        r = player_batches[player][:reward]
        t = player_batches[player][:terminal]

        # for training behavior_actor.
        mu_actions = vcat(
            ((
                player_batches[p][:next_state] |>
                A
            ) for (p, a) in π.agents)...
        )
        # for training behavior_critic.
        new_actions = vcat(
            ((
                player_batches[p][:next_state] |>
                Aₜ
            ) for (p, a) in π.agents)...
        )

        qₜ = Cₜ(vcat(s′, new_actions)) |> vec
        y = r .+ γ .* (1 .- t) .* qₜ

        gs1 = gradient(Flux.params(AC)) do
            q = C(vcat(s, a)) |> vec
            loss = mean((y .- q) .^ 2)
            Flux.ignore_derivatives() do
                p.critic_loss = loss
            end
            loss
        end

        optimise!(AC, gs1)

        gs2 = gradient(Flux.params(AA)) do
            v = C(vcat(s, mu_actions)) |> vec
            reg = mean(A(player_batches[player][:state]) .^ 2)
            loss = -mean(v) + reg * 1e-3
            Flux.ignore_derivatives() do
                p.actor_loss = loss
            end
            loss
        end

        optimise!(AA, gs2)

    end
end