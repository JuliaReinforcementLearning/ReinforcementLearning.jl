abstract type AbstractCFRPolicy <: AbstractPolicy end

function Base.run(
    p::AbstractCFRPolicy,
    env::AbstractEnv,
    stop_condition = StopAfterStep(1),
    hook = EmptyHook(),
)
    @assert NumAgentStyle(env) isa MultiAgent
    @assert DynamicStyle(env) === SEQUENTIAL
    @assert RewardStyle(env) === TERMINAL_REWARD
    @assert ChanceStyle(env) === EXPLICIT_STOCHASTIC
    @assert DefaultStateStyle(env) isa InformationSet

    RLBase.reset!(env)

    while true
        update!(p, env)
        hook(POST_ACT_STAGE, p, env)
        stop_condition(p, env) && break
    end
    update!(p)
end
