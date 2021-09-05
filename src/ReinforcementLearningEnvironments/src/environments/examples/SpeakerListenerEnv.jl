export SpeakerListenerEnv

mutable struct SpeakerListenerEnv{T<:Vector{Float64}} <: AbstractEnv
    target::T
    content::T
    player_vel::T
    player_pos::T
    landmarks_pos::Vector{T}
    landmarks_num::Int
    ϵ
    damping
    max_accel
    space_dim::Int
    init_step::Int
    play_step::Int
    max_steps::Int
    continuous::Bool
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
- `stop = 0.01`, when the distance between the `Listener` and the target is smaller than the `stop`, the game will be terminated.
- `damping = 0.25`, for simulation of the physical space, `Listener`'s action will meet the damping in each step.
- `max_accel = 0.02`, the maximum acceleration of the `Listener` in each step.
- `space_dim::Int = 2`, the dimension of the environment's space.
- `max_steps::Int = 25`, the maximum playing steps in one episode.
- `continuous::Bool = true`, set to `false` if you want the action_space of the players to be discrete. Otherwise, the action_space will be continuous.
"""
function SpeakerListenerEnv(;
    N::Int = 3,
    stop = 0.01,
    damping = 0.25,
    max_accel = 0.5,
    space_dim::Int = 2,
    max_steps::Int = 50,
    continuous::Bool = true)
    SpeakerListenerEnv(
        zeros(N),
        zeros(N),
        zeros(space_dim),
        zeros(space_dim),
        [zeros(space_dim) for _ in Base.OneTo(N)],
        N,
        stop,
        damping,
        max_accel,
        space_dim,
        0,
        0,
        max_steps,
        continuous,
    )
end

function RLBase.reset!(env::SpeakerListenerEnv)
    env.init_step = 0
    env.play_step = 0
    env.target = zeros(env.landmarks_num)
    env.content = zeros(env.landmarks_num)
    env.player_pos = zeros(env.space_dim)
    env.landmarks_pos = [zeros(env.space_dim) for _ in Base.OneTo(env.landmarks_num)]
end

RLBase.is_terminated(env::SpeakerListenerEnv) = (reward(env) > - env.ϵ) || (env.play_step > env.max_steps)
RLBase.players(::SpeakerListenerEnv) = (:Speaker, :Listener, CHANCE_PLAYER)

RLBase.state(env::SpeakerListenerEnv, ::Observation{Any}, players::Tuple) = Dict(p => state(env, p) for p in players)

RLBase.state(env::SpeakerListenerEnv, ::Observation{Any}, player::Symbol) = 
    # for speaker, it can observe the target and help listener to arrive it.
    if player == :Speaker
        env.target
    # for listener, it can observe current velocity, relative positions of landmarks, and speaker's conveyed information.
    elseif player == :Listener
        vcat(
            env.player_vel...,
            (
                vcat((landmark_pos .- env.player_pos)...) for landmark_pos in env.landmarks_pos
            )...,
            env.content...,
        )
    else
        @error "No player $player."
    end

RLBase.state(env::SpeakerListenerEnv, ::Observation{Any}, ::ChancePlayer) = vcat(env.landmarks_pos, [env.player_pos])

RLBase.state_space(env::SpeakerListenerEnv, ::Observation{Any}, players::Tuple) = 
    Space(Dict(player => state_space(env, player) for player in players))

RLBase.state_space(env::SpeakerListenerEnv, ::Observation{Any}, player::Symbol) = 
    if player == :Speaker
        # env.target
        Space([[0., 1.] for _ in Base.OneTo(env.landmarks_num)])
    elseif player == :Listener
        Space(vcat(
            # relative positions of landmarks, no bounds.
            (vcat(
                Space([ClosedInterval(-Inf, Inf) for _ in Base.OneTo(env.space_dim)])...
                ) for _ in Base.OneTo(env.landmarks_num + 1))...,
            # communication content from `Speaker`
            [[0., 1.] for _ in Base.OneTo(env.landmarks_num)],
            ))
    else
        @error "No player $player."
    end

RLBase.state_space(env::SpeakerListenerEnv, ::Observation{Any}, ::ChancePlayer) = 
    Space(
        vcat(
            # landmarks' positions
            (Space([ClosedInterval(-1, 1) for _ in Base.OneTo(env.space_dim)]) for _ in Base.OneTo(env.landmarks_num))...,
            # player's position, no bounds.
            Space([ClosedInterval(-Inf, Inf) for _ in Base.OneTo(env.space_dim)]),
        )
    )

RLBase.action_space(env::SpeakerListenerEnv, players::Tuple) = 
        Space(Dict(p => action_space(env, p) for p in players))

RLBase.action_space(env::SpeakerListenerEnv, player::Symbol) = 
    if player == :Speaker
        env.continuous ? Space([ClosedInterval(0, 1) for _ in Base.OneTo(env.landmarks_num)]) : Space([ZeroTo(1) for _ in Base.OneTo(env.landmarks_num)])
    elseif player == :Listener
        # there has two directions in each dimension.
        env.continuous ? Space([ClosedInterval(0, 1) for _ in Base.OneTo(2 * env.space_dim)]) : Space([ZeroTo(1) for _ in Base.OneTo(2 * env.space_dim)])
    else
        @error "No player $player."
    end

function RLBase.action_space(env::SpeakerListenerEnv, ::ChancePlayer)
    if env.init_step < env.landmarks_num + 1
        Space([ClosedInterval(-1, 1) for _ in Base.OneTo(env.space_dim)])
    else
        Base.OneTo(env.landmarks_num)
    end
end

function (env::SpeakerListenerEnv)(action, ::ChancePlayer)
    env.init_step += 1
    if env.init_step <= env.landmarks_num
        env.landmarks_pos[env.init_step] = action
    elseif env.init_step == env.landmarks_num + 1
        env.player_pos = action
    else
        @assert action in Base.OneTo(env.landmarks_num) "The target should be assigned to one of the landmarks."
        env.target[action] = 1.
    end
end

function (env::SpeakerListenerEnv)(actions::Dict, players::Tuple)
    @assert length(actions) == length(players)
    for p in players
        env(actions[p], p)
    end
    env.play_step += 1
end

function (env::SpeakerListenerEnv)(action::Vector, player::Symbol)
    if player == :Speaker
        # update conveyed information.
        env.content = round.(action)
    elseif player == :Listener
        # update velocity, here env.damping is for simulation physical rule.
        action = round.(action)
        acceleration = [action[2 * i] - action[2 * i - 1] for i in Base.OneTo(env.space_dim)]
        env.player_vel .*= (1 - env.damping)
        env.player_vel .+= (acceleration * env.max_accel)
        # update position
        env.player_pos .+= env.player_vel * 0.1 # velocity * time
    else
        @error "No player $player."
    end
end

RLBase.reward(::SpeakerListenerEnv, ::ChancePlayer) = -Inf

function RLBase.reward(env::SpeakerListenerEnv, p)
    if sum(env.target) == 1
        goal = findfirst(env.target .== 1.)
        -sum((env.landmarks_pos[goal] .- env.player_pos) .^ 2)
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
