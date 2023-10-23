import LogExpFunctions
export CQLSACPolicy

"""
    CQLSACPolicy(
        sac::SACPolicy,
        action_sample_size::Int = 10,
        α_cql::Float32 = 0f0,
        α_lr::Float32 = 1f-3,
        τ_cql::Float32 = 5f0,
        α_cql_autotune::Bool = true,
        cons_weight::Float32 = 1f0, #multiplies the Q difference before substracting tau, used to scale with the rewards.
        finetune_experiment::E = nothing #Provide an second experiment to run at PostExperimentStage to finetune the sac policy, typically with an agent that uses the sac policy. Leave nothing if no finetuning is desired.
    )

    Implements the Conservative Q-Learning algorithm [1] in its continuous variant on top of the SAC algorithm [2]. `CQLSACPolicy` wraps a classic `SACPolicy` whose networks will be trained normally, except for the additional conservative loss.
    CQLSACPolicy contains the additional hyperparameters that are specific to this method. α_cql is the lagrange penalty for the conservative_loss, it will be automatically tuned if ` α_cql_autotune = true`. `cons_weight` is a scaling parameter 
    which may be necessary to decrease if the scale of the Q-values is large. `τ_cql` is the threshold of the lagrange conservative penalty.
    See SACPolicy for all the other hyperparameters related to SAC.

    If desired, you can provide an `Experiment(agent, env, stop_condition, hook)` to finetune_experiment to finish the training with a finetuning run. `agent` should be a normal `Agent` with policy being `sac`, an environment to finetune on. 
    See the example in ReinforcementLearningExperiments.jl for an example on the Pendulum task.
    
    As this is an offline algorithm, it must be wrapped in an `OfflineAgent` which will not update the trajectory as the training progresses. However, it _will_ interact with the supplied environment, which may be useful to record the progress.
    This can be avoided by supplying a dummy environment. 

    [1] Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33, 1179-1191.
    [2] Haarnoja, T. et al. (2018). Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.
"""
Base.@kwdef mutable struct CQLSACPolicy{P<:SACPolicy, E<:Union{Experiment,Nothing}} <: AbstractPolicy
    sac::P
    action_sample_size::Int = 10
    α_cql::Float32 = 0f0
    α_lr::Float32 = 1f-3
    τ_cql::Float32 = 5f0
    α_cql_autotune::Bool = true
    cons_weight::Float32 = 1f0 #multiplies the Q difference before substracting tau, used to scale with the rewards.
    finetune_experiment::E = nothing #Provide an second experiment to run at PostExperimentStage to finetune the sac policy, typically with an agent that uses the sac policy. Leave nothing if no finetuning is desired.
end

function RLBase.plan!(p::CQLSACPolicy, env)
    RLBase.plan!(p.sac, env)
end

function RLBase.optimise!(p::CQLSACPolicy, ::PostActStage, traj::Trajectory)
    batch = ReinforcementLearningTrajectories.StatsBase.sample(traj)
    update_critic!(p, batch)
    update_actor!(p.sac, batch) #uses the implemented SACPolicy actor update, as it is identical
end

function conservative_loss(p::CQLSACPolicy, t_qnetwork, q_policy_inputs, logps, s, a, y)
    qnetwork = model(t_qnetwork)
    q_policy = vec(LogExpFunctions.logsumexp(qnetwork(q_policy_inputs) .- logps, dims = 2)) #(1 x 1 x batchsize) -> (batchsize,) Note: some python public implementations use a temperature.

    q_beta = vec(qnetwork(vcat(s, a))) #(batchsize,)

    diff = mean(q_policy .- q_beta)*p.cons_weight - p.τ_cql

    if p.α_cql_autotune
        p.α_cql += p.α_lr*diff
        p.α_cql = clamp(p.α_cql, 0f0,1f6)
    end

    conservative_loss = p.α_cql*diff 

    q_learning_loss = mse(q_beta, y)

    return conservative_loss + q_learning_loss
end

function update_critic!(p::CQLSACPolicy, batch::NamedTuple{SS′ART})
    s, s′, a, r, t = send_to_device(device(p.sac.qnetwork1), batch)

    y = soft_q_learning_target(p.sac, r, t, s′)

    states = MLUtils.unsqueeze(s, dims = 2) #(state_size x 1 x batchsize)
    a_policy, logp_policy = RLCore.forward(p.sac.policy, states, p.action_sample_size) #(action_size x action_sample_size x batchsize), (1 x action_sample_size x batchsize)
    
    #next_states = MLUtils.unsqueeze(s′, dims = 2) #(state_size x 1 x batchsize)
    #a_policy′, logp_policy′ = RLCore.forward(p.sac.policy, next_states, p.action_sample_size) #(action_size x action_sample_size x batchsize), (1 x action_sample_size x batchsize)
    
    a_unif = (rand(p.sac.rng, Float32, size(a_policy)...) .- 0.5f0) .* 2 # Uniform sampling between -1 and 1: (action_size x action_sample_size x batchsize)
    logp_unif = fill!(similar(a_unif, 1, size(a_unif)[2:end]...), 0.5^size(a_unif)[1]) #(1 x action_sample_size x batchsize) 
    
    repeated_states = reduce(hcat, Iterators.repeated(states, p.action_sample_size*2)) #(state_size x action_sample_size*2 x batchsize)
    actions = hcat(a_policy, a_unif)#, a_policy′) #(action_size x action_sample_size*2 x batchsize)
    
    q_policy_inputs = vcat(repeated_states, actions)
    logps = hcat(logp_policy, logp_unif)#, logp_policy′) #(1 x action_sample_size*2 x batchsize)

    # Train Q Networks
    q_grad_1 = gradient(Flux.params(model(p.sac.qnetwork1))) do
        conservative_loss(p, p.sac.qnetwork1, q_policy_inputs, logps, s, a, y)
    end
    RLBase.optimise!(p.sac.qnetwork1, q_grad_1)

    q_grad_2 = gradient(Flux.params(model(p.sac.qnetwork2))) do
        conservative_loss(p, p.sac.qnetwork2, q_policy_inputs, logps, s, a,y )
    end
    RLBase.optimise!(p.sac.qnetwork2, q_grad_2)
end

RLBase.optimise!(::CQLSACPolicy{<:SACPolicy, Nothing}, ::PostExperimentStage, ::Trajectory) = nothing

function RLBase.optimise!(p::CQLSACPolicy, ::PostExperimentStage, ::Trajectory) 
    println("Finetuning...")
    p.finetune_experiment.policy.trajectory.controller.n_inserted = 0
    p.finetune_experiment.policy.trajectory.controller.n_sampled = 0
    run(p.finetune_experiment)
end
