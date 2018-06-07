using ReinforcementLearning, POMDPModels

import ReinforcementLearning.interact!,
ReinforcementLearning.reset!, ReinforcementLearning.getstate
rng = MersenneTwister()

type POMDPEnvironment
    model::POMDPModels.DiscretePOMDP
    state::Int64
end
type MDPEnvironment
    model::POMDPModels.DiscreteMDP
    state::Int64
end

function interact!(action, env::POMDPEnvironment) 
    s, o, r = generate_sor(env.model, env.state, action, rng)
    env.state = s
    o, r, isterminal(env.model, s)
end
function reset!(env::Union{POMDPEnvironment, MDPEnvironment})
    env.state = initial_state(env.model, rng)
    nothing
end
function getstate(env::POMDPEnvironment)
    generate_o(env.model, env.state, rng),
    isterminal(env.model, env.state)
end

function interact!(action, env::MDPEnvironment)
    s = rand(rng, transition(env.model, env.state, action))
    r = env.model.R[env.state, action]
    env.state = s
    s, r, isterminal(env.model, s)
end
function getstate(env::MDPEnvironment)
    env.state, isterminal(env.model, env.state)
end

