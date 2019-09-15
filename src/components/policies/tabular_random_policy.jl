export TabularRandomPolicy

struct TabularRandomPolicy <: AbstractPolicy
    prob::Array{Float64,2}
end

(π::TabularRandomPolicy)(s) = sample(Weights(π.prob[s, :]))
(π::TabularRandomPolicy)(obs::Observation) = π(get_state(obs))

get_prob(π::TabularRandomPolicy, s) = @view π.prob[s, :]
get_prob(π::TabularRandomPolicy, s, a) = π.prob[s, a]