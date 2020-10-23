using CommonRLInterface: CommonRLInterface

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
    if get_num_players(env) == 1
        convert(CRL.AbstractMarkovEnv, env)
    elseif get_num_players(env) == 2 && UtilityStyle(env) === ZERO_SUM
        convert(CRL.AbstractZeroSumEnv, env)
    else
        CommonRLEnv(env)
    end
end

Base.convert(::Type{CRL.AbstractMarkovEnv}, env::AbstractEnv) = CommonRLMarkovEnv(env)
Base.convert(::Type{CRL.AbstractZeroSumEnv}, env::AbstractEnv) = CommonRLZeroSumEnv(env)

CRL.@provide CRL.reset!(env::CommonRLEnvs) = reset!(env.env)
CRL.@provide CRL.actions(env::CommonRLEnvs) = get_actions(env.env)
CRL.@provide CRL.observe(env::CommonRLEnvs) = get_state(env.env)
CRL.state(env::CommonRLEnvs) = get_state(env.env)
function CRL.provided(::typeof(CRL.state), env::CommonRLEnvs)
    return InformationStyle(env.env) === PERFECT_INFORMATION
end
CRL.@provide CRL.terminated(env::CommonRLEnvs) = get_terminal(env.env)
CRL.@provide CRL.player(env::CommonRLEnvs) = get_current_player(env.env)
CRL.@provide CRL.clone(env::CommonRLEnvs) = CommonRLEnv(copy(env.env))

CRL.@provide function CRL.act!(env::CommonRLEnvs, a)
    env.env(a)
    return get_reward(env.env)
end

CRL.valid_actions(x::CommonRLEnvs) = get_legal_actions(x.env)
function CRL.provided(::typeof(CRL.valid_actions), env::CommonRLEnvs)
    return ActionStyle(env.env) === FullActionSet()
end

CRL.valid_action_mask(x::CommonRLEnvs) = get_legal_actions_mask(x.env)
function CRL.provided(::typeof(CRL.valid_action_mask), env::CommonRLEnvs)
    return ActionStyle(env.env) === FullActionSet()
end

#####
# RLBaseEnv
#####

mutable struct RLBaseEnv{T<:CRL.AbstractEnv,R} <: AbstractEnv
    env::T
    r::R
end

Base.convert(::Type{AbstractEnv}, env::CRL.AbstractEnv) = convert(RLBaseEnv, env)
Base.convert(::Type{RLBaseEnv}, env::CRL.AbstractEnv) = RLBaseEnv(env, 0.0f0)  # can not determine reward ahead. Assume `Float32`.

get_state(env::RLBaseEnv) = CRL.observe(env.env)
get_actions(env::RLBaseEnv) = CRL.actions(env.env)
get_reward(env::RLBaseEnv) = env.r
get_terminal(env::RLBaseEnv) = CRL.terminated(env.env)
get_legal_actions(env::RLBaseEnv) = CRL.valid_actions(env.env)
get_legal_actions_mask(env::RLBaseEnv) = CRL.valid_action_mask(env.env)
reset!(env::RLBaseEnv) = CRL.reset!(env.env)

(env::RLBaseEnv)(a) = env.r = CRL.act!(env.env, a)
Base.copy(env::CommonRLEnv) = RLBaseEnv(CRL.clone(env.env), env.r)

function ActionStyle(env::RLBaseEnv)
    return CRL.provided(CRL.valid_actions, env.env) ? FullActionSet() : MinimalActionSet()
end
UtilityStyle(env::RLBaseEnv) = GENERAL_SUM
UtilityStyle(env::RLBaseEnv{<:CRL.AbstractZeroSumEnv}) = ZERO_SUM
InformationStyle(env::RLBaseEnv) = IMPERFECT_INFORMATION
