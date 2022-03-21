export MPOPolicy
using LinearAlgebra, Flux, Optim
using Zygote: ignore, dropgrad
using ReinforcementLearningCore: logdetLorU

#Note: we use two Q networks, this is not used in the original publications, but there is no reason to not do it since the networks are trained the same way as for example SAC
mutable struct MPOPolicy{P<:NeuralNetworkApproximator,Q<:NeuralNetworkApproximator,R,AV<:AbstractVector} <: AbstractPolicy
    policy::P
    qnetwork1::Q
    qnetwork2::Q
    target_qnetwork1::Q
    target_qnetwork2::Q 
    γ::Float32
    batch_size::Int #N
    action_sample_size::Int #K 
    ϵ::Float32  #KL bound on the non-parametric variational approximation to the policy
    ϵμ::Float32 #KL bound for the parametric policy training of mean estimations
    ϵΣ::Float32 #KL bound for the parametric policy training of (co)variance estimations
    αμ::AV
    αΣ::AV
    update_freq::Int
    update_after::Int
    update_step::Int
    critic_batches::Int
    policy_batches::Int
    τ::Float32 #Polyak avering parameter of target networks
    rng::R
end

function MPOPolicy(;policy::NeuralNetworkApproximator, qnetwork1::Q, qnetwork2::Q, γ = 0.99f0, batch_size, action_sample_size, ϵ = 0.1f0, ϵμ = 5f-4, ϵΣ = 1f-5, update_freq = 1, update_after = 0, critic_batches = 1, policy_batches = 1, τ = 1f-3, rng = Random.GLOBAL_RNG) where Q <: NeuralNetworkApproximator
    @assert device(policy) == device(qnetwork1) == device(qnetwork2) "All network approximators must be on the same device"
    @assert device(policy) == device(rng) "The specified rng does not generate on the same device as the policy. Use `CUDA.CURAND.RNG()` to work with a CUDA GPU"
    αμ = send_to_device(device(policy), [0f0])
    αΣ = send_to_device(device(policy), [0f0])
    MPOPolicy(policy, qnetwork1, qnetwork2, deepcopy(qnetwork1), deepcopy(qnetwork2), γ, batch_size, action_sample_size, ϵ, ϵμ, ϵΣ, αμ, αΣ, update_freq, update_after, 0, critic_batches, policy_batches, τ, rng)
end

Flux.@functor MPOPolicy

function (p::MPOPolicy)(env)
    p.update_step += 1
    D = device(p.policy)
    s = send_to_device(D, state(env))
    action = p.policy(p.rng, s; is_sampling=true)
    send_to_host(action)
end

#Update of the NNs happens here. This function is called at every environment step but will only update every `update_freq` calls. A low update_freq makes for a strongly offpolicy algorithm that will reuse data from far in the past.
#A high `update_freq` will use more recent transitions, but less times. To work only with transitions sampled from the current policy, use a trajectory with a length equal to `update_freq`. If you work with N multiple parallel environment, 
#use `update_freq = length(traj) ÷ N` (remainder should be zero) otherwise some transitions will never be used at all. 

function RLBase.update!(
    p::MPOPolicy,
    traj::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PostActStage
)
    length(traj) > p.update_after || return
    p.update_step % p.update_freq == 0 || return
    for _ in p.critic_batches
        inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
        s, a, r, t, s′ = send_to_device(device(p.qnetwork1), batch)
        update_critic!(p, s, a, r, t, s′)
    end
    for _ in p.policy_batches
        inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
        s = send_to_device(device(p.qnetwork1), first(batch))
        update_policy!(p, s)
    end
    #TODO: make sampler a field of MPOPolicy to accomodate Nstep and Epoch sampling as suits the user.
    #Later I would like this to be 
    #=
    for (inds, batch) in p.trajectory_sampler 
        update!(p, batch)
    end
    =#
end

#Here we apply the TD3 Q network approach. This could be customizable by the user in a new p.critic <: AbstractCritic field. 
function update_critic!(p::MPOPolicy, s, a, r, t, s′)
    γ, τ = p.γ, p.τ

    a′ = p.policy(p.rng, s′; is_sampling=true, is_return_log_prob=false)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec(q′) 

    # Train Q Networks
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
        q1 = p.qnetwork1(q_input) |> vec
        mse(q1, y)
    end
    update!(p.qnetwork1, q_grad_1)
    q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
        q2 = p.qnetwork2(q_input) |> vec
        mse(q2, y)
    end
    update!(p.qnetwork2, q_grad_2)

    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end
end

function update_policy!(p::MPOPolicy, s)
    #Fit non-parametric variational distribution
    states = reshape(s, first(size(s)), 1, :) #3D tensor with dimensions (state_size x 1 x batch_size)
    action_samples, logp_π = p.policy(p.rng, states, p.action_sample_size) #3D tensor with dimensions (action_size x action_sample_size x batchsize)
    input = vcat(repeat(states, outer = (1, p.action_sample_size, 1)), action_samples) #repeat states along 2nd dimension and vcat with sampled actions to get state-action tensor
    Q = p.qnetwork1(input) 
    η = map(q -> solve_mpodual(q, p.ϵ, p.policy), eachslice(send_to_host(Q), dims = 3)) #this must be done on the CPU
    qij = softmax(Q./reshape(η, 1, :, p.batch_size), dims = 2) # dims = (1 x actions_sample_size x batch_size)

    #Improve policy towards qij
    μ_old, L_old = p.policy(p.rng, states, is_sampling = false)
    ps = Flux.params(p.policy, p.αμ, p.αΣ)
    gs = gradient(ps) do 
        loss_decoupled(p, qij, states, action_samples, μ_old, L_old)
    end
    
    if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), gs)
        error("Gradient contains NaN of Inf")
    end

    gs[p.αμ] *= -1 #negative of gradient since we minimize w.r.t. α
    gs[p.αΣ] *= -1 

    Flux.Optimise.update!(p.policy.optimizer, ps, gs)
    p.αμ = clamp.(p.αμ, 0f0, Inf32) #maybe add an upperbound ?
    p.αΣ = clamp.(p.αΣ, 0f0, Inf32)
end

function solve_mpodual(Q, ϵ, nna::NeuralNetworkApproximator)
    solve_mpodual(Q, ϵ, nna.model)
end

function solve_mpodual(Q::AbstractMatrix, ϵ, ::Union{GaussianNetwork, CovGaussianNetwork})
    max_Q = maximum(Q, dims = 1) #needed for numerical stability
    g(η) = only(η .* ϵ .+ mean(max_Q) .+ η .* mean(log.(mean(exp.((Q .- max_Q)./η),dims = 1)),dims = 2))
    η = only(Optim.minimizer(optimize(g, [eps(ϵ)]))) #this uses Nelder-Mead's algorithm, other GD algorithms may be used. Make this a field in MPO struct ?
end

#For CovGaussianNetwork
function loss_decoupled(p::MPOPolicy{<:NeuralNetworkApproximator{<:CovGaussianNetwork}}, qij, states, actions, μ_old, L_old)
    μ, L = p.policy(p.rng, states, is_sampling = false)
    #decoupling
    μ_d, L_d = Zygote.ignore() do 
        μ, L 
    end 
    #decoupled logp for mu and L
    logp_π_new_μ = mvnormlogpdf(μ, L_d, actions) 
    logp_π_new_L = mvnormlogpdf(μ_d, L, actions)
    policy_loss = sum(qij .* (logp_π_new_μ .+ logp_π_new_L))

    μ_old_s, L_old_s, μ_s, L_d_s, μ_d_s, L_s = map(x->eachslice(x, dims =3), (μ_old, L_old, μ, L_d, μ_d, L)) #slice all tensors along 3rd dim
    
    lagrangeμ = only(p.αμ) * (p.ϵμ - mean(mvnorm_kl_divergence.(μ_old_s, L_old_s, μ_s, L_d_s))) 
    lagrangeΣ = only(p.αΣ) * (p.ϵΣ - mean(mvnorm_kl_divergence.(μ_old_s, L_old_s, μ_d_s, L_s)))
    
    return policy_loss + lagrangeμ + lagrangeΣ
end

function mvnorm_kl_divergence(μ1::AbstractMatrix, L1::AbstractMatrix, μ2::AbstractMatrix, L2::AbstractMatrix)
    d = size(μ1,1)
    logdet = logdetLorU(L2) - logdetLorU(L1) 
    trace = tr((L2*L2')\(L1*L1')) # trace of inv(Σ2) * Σ1
    sqmahal = sum(abs2.(L2\(μ2 .- μ1))) #mahalanobis square distance
    return (logdet - d + trace + sqmahal)/2
end

#In the case of diagonal covariance (with GaussianNetwork), 
function loss_decoupled(p::MPOPolicy{<:NeuralNetworkApproximator{<:GaussianNetwork}}, qij, states, actions, μ_old, σ_old)
    μ, logσ = p.policy(p.rng, states, is_sampling = false) #3D tensors with dimensions (action_size x 1 x batch_size)
    σ = exp.(logσ)
    μ_d, σ_d = Zygote.ignore() do
        μ, σ #decoupling
    end
    #decoupled logp for mu and sigma
    logp_π_new_μ = sum(normlogpdf(μ, σ_d, actions) .- (2.0f0 .* (log(2.0f0) .- actions .- softplus.(-2.0f0 .* actions))), dims = 1)
    logp_π_new_σ = sum(normlogpdf(μ_d, σ, actions) .- (2.0f0 .* (log(2.0f0) .- actions .- softplus.(-2.0f0 .* actions))), dims = 1)
    policy_loss = sum(qij .* (logp_π_new_μ .+ logp_π_new_σ))
    μ_old_s, σ_old_s, μ_s, σ_d_s, μ_d_s, σ_s = map(x->eachslice(x, dims =3), (μ_old, σ_old, μ, σ_d, μ_d, σ)) #slice all tensors along 3rd dim
    lagrangeμ = only(p.αμ) * (p.ϵμ - mean(norm_kl_divergence.(μ_old_s, σ_old_s, μ_s, σ_d_s))) 
    lagrangeΣ = only(p.αΣ) * (p.ϵΣ - mean(norm_kl_divergence.(μ_old_s, σ_old_s, μ_d_s, σ_s)))
    
    return policy_loss + lagrangeμ + lagrangeΣ
    
    return policy_loss + lagrangeloss
end

#computes the KL divergence between two multivariate gaussians with diagonal covariances. Parameters must be input as Matrices
function norm_kl_divergence(μ1::AbstractVecOrMat, σ1::AbstractVecOrMat, μ2::AbstractVecOrMat, σ2::AbstractVecOrMat)
    d = size(μ1,1)
    logdet = sum(log.(σ2)) - sum(log.(σ1)) 
    trace = sum(σ1 ./ σ2)
    sqmahal = sum((μ2 .- μ1) .^2 ./ σ2)
    return (logdet - d + trace + sqmahal)/2
end

#TODO: handle Categorical actor. Add a CategoricalNetwork that has the same api than Gaussians ?

#=
    Flux.gpu(p::MPOPolicy; rng = CUDA.CURAND.RNG())

Apply Flux.gpu to all neural nets of the policy. 
`rng` can be used to specificy a particular rng if desired, make sure this rng generates numbers on the correct device.
=#
function Flux.gpu(p::MPOPolicy; rng = CUDA.CURAND.RNG())
    MPOPolicy(
        p.policy |> gpu, 
        p.qnetwork1 |> gpu, 
        p.qnetwork2 |> gpu, 
        p.target_qnetwork1 |> gpu, 
        p.target_qnetwork2 |> gpu, 
        p.γ, 
        p.batch_size, 
        p.action_sample_size, 
        p.ϵ, 
        p.ϵμ, 
        p.ϵΣ, 
        p.αμ |> gpu, 
        p.αΣ |> gpu, 
        p.update_freq, 
        p.update_after, 
        p.update_step,
        p.critic_batches, 
        p.policy_batches, 
        p.τ, 
        rng)
end

#=
    send_to_device(device, p::MPOPolicy; rng = device isa CuDevice ? CUDA.CURAND.RNG() : GLOBAL_RNG)

Send all neural nets of the policy to a specified device.
`rng` can be used to specificy a particular rng if desired, make sure this rng generates numbers on `device`. 
=#
function send_to_device(device, p::MPOPolicy; rng = device isa CuDevice ? CUDA.CURAND.RNG() : GLOBAL_RNG)
    sd(x) = send_to_device(device, x) 
    MPOPolicy(
        p.policy |> sd, 
        p.qnetwork1 |> sd, 
        p.qnetwork2 |> sd, 
        p.target_qnetwork1 |> sd, 
        p.target_qnetwork2 |> sd, 
        p.γ, 
        p.batch_size, 
        p.action_sample_size, 
        p.ϵ, 
        p.ϵμ, 
        p.ϵΣ, 
        p.αμ |> sd, 
        p.αΣ |> sd, 
        p.update_freq, 
        p.update_after, 
        p.update_step,
        p.critic_batches, 
        p.policy_batches, 
        p.τ, 
        rng)
end

#=
    send_to_host(p::MPOPolicy; rng = GLOBAL_RNG)

Send all neural nets of the policy to the cpu.
`rng` can be used to specificy a particular rng if desired. 
=#
function send_to_host(p::MPOPolicy; rng = GLOBAL_RNG)
    send_to_device(Val{:cpu}, p, rng = rng)
end