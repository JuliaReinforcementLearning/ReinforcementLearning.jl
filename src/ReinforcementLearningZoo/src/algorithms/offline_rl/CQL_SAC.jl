import LogExpFunctions
export CQLSACPolicy

Base.@kwdef mutable struct CQLSACPolicy{P<:SACPolicy} <: AbstractPolicy
    sac::P
    action_sample_size::Int = 10
    α_cql::Float32 = 0f0
    α_lr::Float32 = 1f-3
    τ_cql::Float32 = 5f0
    α_cql_autotune::Bool = true
    cons_weight::Float32 = 1f0 #multiplies the Q difference before substracting tau, used to scale with the rewards.
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