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
    policy::Tp = defaultpolicy(learner, environment.actionspace, buffer)
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
    statetype = typeof(preprocessstate(preprocessor, getstate(env)[1]))
    state_sz = size(preprocessstate(preprocessor, getstate(env)[1]))
    actiontype = typeof(sample(actionspace(env)))
    action_sz = size(sample(actionspace(env)))
    if capacity < 0
        EpisodeTurnBuffer{statetype, actiontype, Float64, Bool}()
    else
        CircularTurnBuffer{statetype, actiontype, Float64, Bool}(capacity, state_sz, action_sz, (), ())
    end
end

