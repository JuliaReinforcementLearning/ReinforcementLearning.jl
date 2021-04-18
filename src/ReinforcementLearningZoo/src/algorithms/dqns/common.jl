#####
# some common components for Prioritized Experience Replay based methods
#####

const PERLearners = Union{PrioritizedDQNLearner,RainbowLearner,IQNLearner}

function RLBase.update!(learner::Union{DQNLearner,REMDQNLearner,PERLearners}, t::AbstractTrajectory)
    length(t[:terminal]) - learner.sampler.n <= learner.min_replay_history && return

    learner.update_step += 1

    if learner.update_step % learner.target_update_freq == 0
        copyto!(learner.target_approximator, learner.approximator)
    end

    learner.update_step % learner.update_freq == 0 || return

    inds, batch = sample(learner.rng, t, learner.sampler)

    if t isa PrioritizedTrajectory
        priorities = update!(learner, batch)
        t[:priority][inds] .= priorities
    else
        update!(learner, batch)
    end
end

function RLBase.update!(
    trajectory::PrioritizedTrajectory,
    p::QBasedPolicy{<:PERLearners},
    env::AbstractEnv,
    ::PostActStage,
)
    push!(trajectory[:reward], reward(env))
    push!(trajectory[:terminal], is_terminated(env))
    push!(trajectory[:priority], p.learner.default_priority)
end

"""
    DuelingNetwork(;base, val, adv)
    
Dueling network automatically produces separate estimates of the state value function network and advantage function network. The expected output size of val is 1, and adv is the size of the action space.
"""
struct DuelingNetwork{B,V,A}
    base::B
    val::V
    adv::A
end

Flux.@functor DuelingNetwork

function (m::DuelingNetwork)(state)
    x = m.base(state)
    val = m.val(x)
    return val .+ m.adv(x) .- mean(m.adv(x), dims=1)
end