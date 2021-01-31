export RandomStartPolicy

Base.@kwdef mutable struct RandomStartPolicy{P,R<:RandomPolicy} <: AbstractPolicy
    policy::P
    random_policy::R
    num_rand_start::Int
end

function (p::RandomStartPolicy)(env)
    p.num_rand_start -= 1
    if p.num_rand_start < 0
        p.policy(env)
    else
        p.random_policy(env)
    end
end

function RLBase.update!(
    p::RandomStartPolicy,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    update!(p.policy, t, e, s)
end

for f in (:prob, :priority)
    @eval function RLBase.$f(p::RandomStartPolicy, args...)
        if p.num_rand_start < 0
            $f(p.policy, args...)
        else
            $f(p.random_policy, args...)
        end
    end
end
