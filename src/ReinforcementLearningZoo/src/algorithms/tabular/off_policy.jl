export OffPolicy, VectorWSARTTrajectory

Base.@kwdef struct OffPolicy{P,B} <: AbstractPolicy
    π_target::P
    π_behavior::B
end

(π::OffPolicy)(env) = π.π_behavior(env)

const VectorWSARTTrajectory = Trajectory{<:NamedTuple{(:weight, SART...)}}

function VectorWSARTTrajectory(;
    weight = Float64,
    state = Int,
    action = Int,
    reward = Float32,
    terminal = Bool,
)
    VectorTrajectory(;
        weight = Float64,
        state = state,
        action = action,
        reward = reward,
        terminal = terminal,
    )
end

Base.length(t::VectorWSARTTrajectory) = length(t[:terminal])

function RLBase.update!(
    p::OffPolicy,
    t::VectorWSARTTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    update!(p.π_target, t, e, s)
end

function RLBase.update!(
    t::VectorWSARTTrajectory,
    p::OffPolicy,
    env::AbstractEnv,
    ::PreActStage,
    a,
)
    s = state(env)
    push!(t[:state], s)
    push!(t[:action], a)

    w = prob(p.π_target, env, a) / prob(p.π_behavior, env, a)
    push!(t[:weight], w)
end

function RLBase.update!(
    t::VectorWSARTTrajectory,
    p::Union{
        OffPolicy{<:QBasedPolicy{<:TDLearner}},
        OffPolicy{<:VBasedPolicy{<:MonteCarloLearner}},
    },
    env::AbstractEnv,
    s::PreEpisodeStage,
)
    empty!(t)
end

function RLBase.update!(
    t::VectorWSARTTrajectory,
    p::Union{
        OffPolicy{<:QBasedPolicy{<:TDLearner}},
        OffPolicy{<:VBasedPolicy{<:MonteCarloLearner}},
    },
    env::AbstractEnv,
    s::PostEpisodeStage,
)
    action = rand(action_space(env))

    push!(t[:state], state(env))
    push!(t[:action], action)
    push!(t[:weight], 1.0)
end
