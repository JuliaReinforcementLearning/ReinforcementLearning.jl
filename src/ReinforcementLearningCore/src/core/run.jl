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

Base.show(io::IO, m::MIME"text/plain", t::Experiment{S}) where {S} = show(io, m, convert(AnnotatedStructTree, t; description=S))

Base.run(ex::Experiment) = run(ex.policy, ex.env, ex.stop_condition, ex.hook)

function Base.run(
    policy::AbstractPolicy,
    env::AbstractEnv,
    stop_condition=StopAfterEpisode(1),
    hook=EmptyHook(),
    reset_condition=ResetAtTerminal()
)
    policy, env = check(policy, env)
    _run(policy, env, stop_condition, hook, reset_condition)
end

"Inject some customized checkings here by overwriting this function"
check(policy, env) = policy, env

function _run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook, reset_condition)

    hook(PreExperimentStage(), policy, env)
    policy(PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        reset!(env)
        policy(PreEpisodeStage(), env)
        hook(PreEpisodeStage(), policy, env)

        while !reset_condition(policy, env) # one episode
            policy(PreActStage(), env)
            hook(PreActStage(), policy, env)

            env |> policy |> env
            optimise!(policy)

            policy(PostActStage(), env)
            hook(PostActStage(), policy, env)

            if stop_condition(policy, env)
                is_stop = true
                policy(PreActStage(), env)
                hook(PreActStage(), policy, env)
                policy(env)  # let the policy see the last observation
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(PostEpisodeStage(), env)  # let the policy see the last observation
            hook(PostEpisodeStage(), policy, env)
        end
    end
    policy(PostExperimentStage(), env)
    hook(PostExperimentStage(), policy, env)
    hook
end
