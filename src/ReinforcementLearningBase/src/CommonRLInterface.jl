import CommonRLInterface

const CRL = CommonRLInterface

#####
# CommonRLEnv
#####

struct CommonRLEnv{T<:AbstractEnv} <: CRL.AbstractEnv
    env::T
end

function Base.convert(::Type{CRL.AbstractEnv}, env::AbstractEnv)
    CommonRLEnv(env)
end

CRL.@provide CRL.reset!(env::CommonRLEnv) = reset!(env.env)
CRL.@provide CRL.actions(env::CommonRLEnv) = action_space(env.env)
CRL.@provide CRL.terminated(env::CommonRLEnv) = is_terminated(env.env)

CRL.@provide function CRL.act!(env::CommonRLEnv, a)
    env.env(a)
    reward(env.env)
end

function find_state_style(env::AbstractEnv, s)
    find_state_style(StateStyle(env), s)
end

find_state_style(::Tuple{}, s) = nothing

function find_state_style(ss::Tuple, s)
    x = first(ss)
    if x isa s
        x
    else
        find_state_style(Base.tail(ss), s)
    end
end

# !!! may need to be extended by user
CRL.@provide CRL.observe(env::CommonRLEnv) = state(env.env)

CRL.provided(::typeof(CRL.state), env::CommonRLEnv) = !isnothing(find_state_style(env.env, InternalState))
CRL.state(env::CommonRLEnv) = state(env.env, find_state_style(env.env, InternalState))

CRL.@provide CRL.clone(env::CommonRLEnv) = CommonRLEnv(copy(env.env))
CRL.@provide CRL.render(env::CommonRLEnv) = @error "unsupported yet..."
CRL.@provide CRL.player(env::CommonRLEnv) = current_player(env.env)

CRL.valid_actions(x::CommonRLEnv) = legal_action_space(x.env)
CRL.provided(::typeof(CRL.valid_actions), env::CommonRLEnv) =
    ActionStyle(env.env) === FullActionSet()

CRL.valid_action_mask(x::CommonRLEnv) = legal_action_space_mask(x.env)
CRL.provided(::typeof(CRL.valid_action_mask), env::CommonRLEnv) =
    ActionStyle(env.env) === FullActionSet()

CRL.@provide CRL.observations(env::CommonRLEnv) = state_space(env.env)

#####
# RLBaseEnv
#####

mutable struct RLBaseEnv{T<:CRL.AbstractEnv,R} <: AbstractEnv
    env::T
    r::R
end

Base.convert(::Type{AbstractEnv}, env::CRL.AbstractEnv) = convert(RLBaseEnv, env)
Base.convert(::Type{RLBaseEnv}, env::CRL.AbstractEnv) = RLBaseEnv(env, 0.0f0)  # can not determine reward ahead. Assume `Float32`.

RLBase.StateStyle(env::RLBaseEnv) = (
    (CRL.provided(CRL.observe, env.env) ? (Observation{Any}(),) : ())...,
    (CRL.provided(CRL.state, env.env) ? (InternalState{Any}(),) : ())...,
)

state(env::RLBaseEnv, ::Observation) = CRL.observe(env.env)
state(env::RLBaseEnv, ::InternalState) = CRL.state(env.env)

state_space(env::RLBaseEnv, ::Observation) = CRL.observations(env.env)

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

current_player(env::RLBaseEnv) = CRL.player(env.env)
players(env::RLBaseEnv) = CRL.players(env.env)