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

RLBase.update!(p::RandomStartPolicy, experience) = update!(p.policy, experience)

for f in (:prob, :priority)
    @eval function RLBase.$f(p::RandomStartPolicy, args...)
        if p.num_rand_start < 0
            $f(p.policy, args...)
        else
            $f(p.random_policy, args...)
        end
    end
end
