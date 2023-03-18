struct SRT{S,R,T}
    state::S
    reward::R
    terminal::T

    function SRT()
        new{Nothing, Nothing, Nothing}(nothing, nothing, nothing)
    end

    function SRT{S,R,T}(state::S, reward::R, terminal::T) where {S,R,T}
        new{S,R,T}(state, reward, terminal)
    end

    function SRT(srt::SRT{Nothing,R,T}, state::S) where {S,R,T}
        new{S,R,T}(state, srt.reward, srt.terminal)
    end

    function SRT(srt::SRT{Nothing,Nothing,Nothing}, state::S) where {S}
        new{S,Nothing,Nothing}(state, srt.reward, srt.terminal)
    end
end

Base.push!(t::Trajectory, srt::SRT) = throw(ArgumentError("action must be supplied when pushing SRT to trajectory"))

function Base.push!(t::Trajectory, srt::SRT{S,Nothing,Nothing}, action::A) where {S,A}
    push!(t, @NamedTuple{state::S, action::A}((srt.state, action)))
end

function Base.push!(t::Trajectory, srt::SRT{S,R,T}, action::A) where {S,A,R,T}
    push!(t, @NamedTuple{state::S, action::A, reward::R, terminal::T}((srt.state, action, srt.reward, srt.terminal)))
end

Base.isempty(srt::SRT{Nothing,Nothing,Nothing}) = true
Base.isempty(srt::SRT) = false

function update!(agent::Agent{P,Tr,SRT}, state::S) where {P <: AbstractPolicy, Tr <: Trajectory, S}
    agent.cache = SRT(agent.cache, state)
end

function (agent::Agent)(::PostActStage, env::E) where {E <: AbstractEnv}
    agent.cache = SRT{Nothing, Any, Bool}(nothing, reward(env), is_terminated(env))
end

function (agent::Agent)(::PostExperimentStage, env::E) where {E <: AbstractEnv}
    agent.cache = SRT()
end
