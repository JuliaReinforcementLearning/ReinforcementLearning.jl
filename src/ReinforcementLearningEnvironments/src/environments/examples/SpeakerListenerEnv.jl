export SpeakerListenerEnv

mutable struct SpeakerListenerEnv <: AbstractEnv
    target::Int
    content::Int
    player_vel::Vector{Float64}
    player_pos::Vector{Float64}
    landmarks_pos::Vector{Vector{Float64}}
    landmarks_num::Int
    ε::Float64 
    damping::Float64
    space_dim::Int
    init_step::Int
    play_step::Int
    max_steps::Int
end

"""
    SpeakerListenerEnv(;kwargs...)

`SpeakerListenerEnv` is a simple cooperative environment of two agents, a `Speaker` and a `Listener`, who are placed in an environment
with `N` landmarks. At each episode, the `Listener` must navigate to a particular landmark(`env.target`) and obtains **reward** based 
on its distance to the target. However, while the `Listener` can observe the relative position of the landmarks, it doesn't know which 
is the target landmark. Conversely, the `Speaker` can observe the target's landmark, and it can produce a communication output(`env.content`) 
at each time step which is observed by the `Listener`.

For more concrete description, you can refer to:
* [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
* [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs)

# Keyword arguments
- `N::Int = 3`, the number of landmarks in the environment.
- `stop::Float64 = 0.01`, when the distance between the `Listener` and the target is smaller than `env.stop`, the game is terminated.
- `space_dim::Int = 2`, the dimension of the environment's space.
- `max_steps::Int = 25`, the maximum playing steps in one episode.
"""
function SpeakerListenerEnv(;
    N::Int = 3,
    stop::Float64 = 0.01,
    space_dim::Int = 2,
    max_steps::Int = 50)
    SpeakerListenerEnv(
        0,
        0,        
        zeros(space_dim),
        zeros(space_dim),
        [zeros(space_dim) for _ in Base.OneTo(N)],
        N,
        stop,
        0.25,
        space_dim,
        0,
        0,
        max_steps,
    )
end

generate_space(env::SpeakerListenerEnv, low::T, high::T) where T<:Float64 = 
    env.space_dim == 1 ? ClosedInterval{T}(low, high) : 
    Space([ClosedInterval{T}(low, high) for _ in Base.OneTo(env.space_dim)])

function RLBase.reset!(env::SpeakerListenerEnv)
    env.target = 0
    env.content = 0
    env.init_step = 0
    env.play_step = 0
    env.player_pos = zeros(env.space_dim)
    env.landmarks_pos = [zeros(env.space_dim) for _ in Base.OneTo(env.landmarks_num)]
end

RLBase.is_terminated(env::SpeakerListenerEnv) = (reward(env) > - env.ε) || (env.play_step > env.max_steps)
RLBase.players(env::SpeakerListenerEnv) = (:Speaker, :Listener, CHANCE_PLAYER)

RLBase.state(env::SpeakerListenerEnv, ::Observation{Any}, players::Tuple) = Dict(p => state(env, p) for p in players)

RLBase.state(env::SpeakerListenerEnv, ::Observation{Any}, player::Symbol) = 
    # for speaker, it can observe the target and help listener to arrive it.
    if player == :Speaker
        [Float64(env.target)]
    # for listener, it can observe current velocity, relative positions of landmarks, and speaker's conveyed information.
    elseif player == :Listener
        vcat(
            env.player_vel...,
            (
                vcat((landmark_pos .- env.player_pos)...) for landmark_pos in env.landmarks_pos
            )...,
            env.content,
        )
    else
        @error "No player $player."
    end

RLBase.state(env::SpeakerListenerEnv, ::Observation{Any}, ::ChancePlayer) = vcat(env.landmarks_pos, [env.player_pos])

RLBase.state_space(env::SpeakerListenerEnv, ::Observation{Any}, players::Tuple) = 
    Space(Dict(player => state_space(env, player) for player in players))

RLBase.state_space(env::SpeakerListenerEnv, ::Observation{Any}, player::Symbol) = 
    if player == :Speaker
        [[Float64(i)] for i in ZeroTo(env.landmarks_num)]
    elseif player == :Listener
        Space(vcat(
            (vcat(generate_space(env, -Inf, Inf)...) for _ in Base.OneTo(env.landmarks_num + 1))...,
            [ZeroTo(env.landmarks_num + 1)],
            ))
    else
        @error "No player $player."
    end

RLBase.state_space(env::SpeakerListenerEnv, ::Observation{Any}, ::ChancePlayer) = 
    Space(
        vcat(
            (generate_space(env, -1., 1.) for _ in Base.OneTo(env.landmarks_num))...,
            generate_space(env, -Inf, Inf),
        )
    )

RLBase.action_space(env::SpeakerListenerEnv, players::Tuple) = 
        Space(Dict(p => action_space(env, p) for p in players))

RLBase.action_space(env::SpeakerListenerEnv, player::Symbol) = 
    if player == :Speaker
        ClosedInterval(-env.landmarks_num, env.landmarks_num)
    elseif player == :Listener
        generate_space(env, -0.02, 0.02)
    else
        @error "No player $player."
    end

function RLBase.action_space(env::SpeakerListenerEnv, ::ChancePlayer)
    if env.init_step <= env.landmarks_num
        generate_space(env, -1., 1.)
    else
        Base.OneTo(env.landmarks_num)
    end
end

function (env::SpeakerListenerEnv)(action, ::ChancePlayer)
    env.init_step += 1
    if env.init_step <= env.landmarks_num
        env.landmarks_pos[env.init_step] .= action
    elseif env.init_step == env.landmarks_num + 1
        env.player_pos .= action
    else
        @assert action in Base.OneTo(env.landmarks_num) "The target should be assigned to one of the landmarks."
        env.target = action
    end
end

function (env::SpeakerListenerEnv)(actions::Dict, players::Tuple)
    @assert length(actions) == length(players)
    for p in players
        env(actions[p], p)
    end
    env.play_step += 1
end

function (env::SpeakerListenerEnv)(action::Union{Float64, Vector{Float64}}, player::Symbol)
    if player == :Speaker
        # update conveyed information.
        env.content = Int(ceil(abs(action)))
    elseif player == :Listener
        @assert length(action) == env.space_dim "The dimension of Listener's action should be the same as position."
        # update velocity, here env.damping is for simulation physical rule.
        env.player_vel .*= (1 - env.damping)
        env.player_vel .+= action
        # update position
        env.player_pos .+= env.player_vel
    else
        @error "No player $player."
    end
end

RLBase.reward(env::SpeakerListenerEnv, ::ChancePlayer) = -Inf

function RLBase.reward(env::SpeakerListenerEnv, p)
    if env.target in Base.OneTo(env.landmarks_num)
        -sqrt(sum((env.landmarks_pos[env.target] .- env.player_pos) .^ 2))
    else
        -Inf
    end
end

RLBase.current_player(env::SpeakerListenerEnv) = 
    if env.init_step < env.landmarks_num + 2
        CHANCE_PLAYER
    else
        (:Speaker, :Listener)
    end

RLBase.NumAgentStyle(::SpeakerListenerEnv) = MultiAgent(2)
RLBase.DynamicStyle(::SpeakerListenerEnv) = SIMULTANEOUS
RLBase.ActionStyle(::SpeakerListenerEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::SpeakerListenerEnv) = IMPERFECT_INFORMATION
RLBase.ChanceStyle(::SpeakerListenerEnv) = EXPLICIT_STOCHASTIC
