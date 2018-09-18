export RLSetup

"""
    @with_kw mutable struct RLSetup{Tl,Tb,Tp,Tpp,Te,Ts}
        learner::Tl
        environment::Te
        stoppingcriterion::Ts
        preprocessor::Tpp = NoPreprocessor()
        buffer::Tb = defaultbuffer(learner, environment, preprocessor)
        policy::Tp = defaultpolicy(learner, environment.actionspace, buffer)
        callbacks::Array{Any, 1} = []
        islearning::Bool = true
        fillbuffer::Bool = islearning
"""
@with_kw mutable struct RLSetup{Tl,Tb,Tp,Tpp,Te,Ts}
    learner::Tl
    environment::Te
    stoppingcriterion::Ts
    preprocessor::Tpp = NoPreprocessor()
    buffer::Tb = defaultbuffer(learner, environment, preprocessor)
    policy::Tp = defaultpolicy(learner, actionspace(environment), buffer)
    callbacks::Array{Any, 1} = []
    islearning::Bool = true
    fillbuffer::Bool = islearning
end

"""
    RLSetup(learner, env, stop; kargs...) = RLSetup(learner = learner,
                                                    environment = env,
                                                    stoppingcriterion = stop;
                                                    kargs...)
"""
RLSetup(learner, env, stop; kargs...) = RLSetup(learner = learner,
                                                environment = env,
                                                stoppingcriterion = stop;
                                                kargs...)
function defaultbuffer(learner, env, preprocessor)
    capacity = :nsteps in fieldnames(typeof(learner)) ? learner.nsteps : 1
    statetype = typeof(preprocessstate(preprocessor, getstate(env).observation))
    state_sz = size(preprocessstate(preprocessor, getstate(env).observation))
    actiontype = typeof(sample(actionspace(env)))
    if capacity < 0
        EpisodeTurnBuffer{Turn{statetype, actiontype, Float64, Bool}}()
    else
        CircularTurnBuffer{Turn{statetype, actiontype, Float64, Bool}}(capacity, state_sz)
    end
end

