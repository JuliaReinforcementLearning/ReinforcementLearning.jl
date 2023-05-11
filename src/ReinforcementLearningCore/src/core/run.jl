export @E_cmd, Experiment


import Parsers

macro E_cmd(s)
    Experiment(s)
end

function try_parse(s, TS=(Bool, Int, Float32, Float64))
    if s == "nothing"
        nothing
    else
        for T in TS
            res = Parsers.tryparse(T, s)
            if !isnothing(res)
                return res
            end
        end
        s
    end
end

function try_parse_kw(s)
    kw = []
    # !!! obviously, it's not correct when a value is string and contains ","
    for part in split(s, ",")
        kv = split(part, "=")
        @assert length(kv) == 2
        k, v = kv
        push!(kw, Symbol(strip(k)) => try_parse(strip(v)))
    end
    NamedTuple(kw)
end

struct Experiment{S}
    policy::Any
    env::Any
    stop_condition::Any
    hook::Any
end

Experiment(args...) = Experiment{Symbol()}(args...)

function Experiment(s::String)
    m = match(r"(?<source>\w+)_(?<method>\w+)_(?<env>\w+)(\((?<game>.*)\))?", s)
    isnothing(m) && throw(
        ArgumentError(
            "invalid format, got $s, expected format is JuliaRL_DQN_Atari(game=\"pong\")`",
        ),
    )
    source = m[:source]
    method = m[:method]
    env = m[:env]
    kw_args = isnothing(m[:game]) ? (;) : try_parse_kw(m[:game])
    ex = Experiment(Val(Symbol(source)), Val(Symbol(method)), Val(Symbol(env)); kw_args...)
    Experiment{Symbol(s)}(ex.policy, ex.env, ex.stop_condition, ex.hook)
end

Base.show(io::IO, m::MIME"text/plain", t::Experiment{S}) where {S} = show(io, m, convert(AnnotatedStructTree, t; description=string(S)))

function Base.run(ex::Experiment)
    run(ex.policy, ex.env, ex.stop_condition, ex.hook)
    return ex
end

function Base.run(
    policy::AbstractPolicy,
    env::AbstractEnv,
    stop_condition::AbstractStopCondition=StopAfterEpisode(1),
    hook::AbstractHook=EmptyHook(),
    reset_condition::AbstractResetCondition=ResetAtTerminal()
)
    policy, env = check(policy, env)
    _run(policy, env, stop_condition, hook, reset_condition)
end

"Inject some customized checkings here by overwriting this function"
check(policy, env) = policy, env

function _run(policy::AbstractPolicy,
        env::AbstractEnv,
        stop_condition::AbstractStopCondition,
        hook::AbstractHook,
        reset_condition::AbstractResetCondition)
    update!(hook, PreExperimentStage(), policy, env)
    update!(policy, PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        reset!(env)
        update!(policy, PreEpisodeStage(), env)
        update!(hook, PreEpisodeStage(), policy, env)

        while !reset_condition(policy, env) # one episode
            update!(policy, PreActStage(), env)
            update!(hook, PreActStage(), policy, env)

            action = RLBase.plan!(policy, env)
            act!(env, action)

            optimise!(policy)

            update!(policy, PostActStage(), env)
            update!(hook, PostActStage(), policy, env)

            if stop_condition(policy, env)
                is_stop = true
                update!(policy, PreActStage(), env)
                update!(hook, PreActStage(), policy, env)
                RLBase.plan!(policy, env)  # let the policy see the last observation
                break
            end
        end # end of an episode

        if is_terminated(env)
            update!(policy, PostEpisodeStage(), env)  # let the policy see the last observation
            update!(hook, PostEpisodeStage(), policy, env)
        end
    end
    update!(policy, PostExperimentStage(), env)
    update!(hook, PostExperimentStage(), policy, env)
    hook
end
