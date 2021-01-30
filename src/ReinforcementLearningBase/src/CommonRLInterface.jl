import CommonRLInterface

const CRL = CommonRLInterface

#####
# CommonRLEnv
#####

struct CommonRLEnv{T<:AbstractEnv} <: CRL.AbstractEnv
    env::T
end

struct CommonRLMarkovEnv{T<:AbstractEnv} <: CRL.AbstractMarkovEnv
    env::T
end

struct CommonRLZeroSumEnv{T<:AbstractEnv} <: CRL.AbstractZeroSumEnv
    env::T
end

const CommonRLEnvs = Union{CommonRLEnv,CommonRLMarkovEnv,CommonRLZeroSumEnv}

function Base.convert(::Type{CRL.AbstractEnv}, env::AbstractEnv)
    if NumAgentStyle(env) === SINGLE_AGENT
        convert(CRL.AbstractMarkovEnv, env)
    elseif NumAgentStyle(env) isa MultiAgent{2} && UtilityStyle(env) === ZERO_SUM
        convert(CRL.AbstractZeroSumEnv, env)
    else
        CommonRLEnv(env)
    end
end

Base.convert(::Type{CRL.AbstractMarkovEnv}, env::AbstractEnv) = CommonRLMarkovEnv(env)
Base.convert(::Type{CRL.AbstractZeroSumEnv}, env::AbstractEnv) = CommonRLZeroSumEnv(env)

CRL.@provide CRL.reset!(env::CommonRLEnvs) = reset!(env.env)
CRL.@provide CRL.actions(env::CommonRLEnvs) = action_space(env.env)
CRL.@provide CRL.observe(env::CommonRLEnvs) = state(env.env)
CRL.state(env::CommonRLEnvs) = state(env.env)
CRL.provided(::typeof(CRL.state), env::CommonRLEnvs) =
    InformationStyle(env.env) === PERFECT_INFORMATION
CRL.@provide CRL.terminated(env::CommonRLEnvs) = is_terminated(env.env)
CRL.@provide CRL.player(env::CommonRLEnvs) = current_player(env.env)
CRL.@provide CRL.clone(env::CommonRLEnvs) = CommonRLEnv(copy(env.env))

CRL.@provide function CRL.act!(env::CommonRLEnvs, a)
    env.env(a)
    reward(env.env)
end

CRL.valid_actions(x::CommonRLEnvs) = legal_action_space(x.env)
CRL.provided(::typeof(CRL.valid_actions), env::CommonRLEnvs) =
    ActionStyle(env.env) === FullActionSet()

CRL.valid_action_mask(x::CommonRLEnvs) = legal_action_space_mask(x.env)
CRL.provided(::typeof(CRL.valid_action_mask), env::CommonRLEnvs) =
    ActionStyle(env.env) === FullActionSet()

#####
# RLBaseEnv
#####

mutable struct RLBaseEnv{T<:CRL.AbstractEnv,R} <: AbstractEnv
    env::T
    r::R
end

Base.convert(::Type{AbstractEnv}, env::CRL.AbstractEnv) = convert(RLBaseEnv, env)
Base.convert(::Type{RLBaseEnv}, env::CRL.AbstractEnv) = RLBaseEnv(env, 0.0f0)  # can not determine reward ahead. Assume `Float32`.

state(env::RLBaseEnv) = CRL.observe(env.env)
state_space(env::RLBaseEnv) = CRL.observations(env.env)
action_space(env::RLBaseEnv) = CRL.actions(env.env)
reward(env::RLBaseEnv) = env.r
is_terminated(env::RLBaseEnv) = CRL.terminated(env.env)
legal_action_space(env::RLBaseEnv) = CRL.valid_actions(env.env)
legal_action_space_mask(env::RLBaseEnv) = CRL.valid_action_mask(env.env)
reset!(env::RLBaseEnv) = CRL.reset!(env.env)

(env::RLBaseEnv)(a) = env.r = CRL.act!(env.env, a)
Base.copy(env::CommonRLEnv) = RLBaseEnv(CRL.clone(env.env), env.r)

ActionStyle(env::RLBaseEnv) =
    CRL.provided(CRL.valid_actions, env.env) ? FullActionSet() : MinimalActionSet()
UtilityStyle(env::RLBaseEnv) = GENERAL_SUM
UtilityStyle(env::RLBaseEnv{<:CRL.AbstractZeroSumEnv}) = ZERO_SUM
InformationStyle(env::RLBaseEnv) = IMPERFECT_INFORMATION
