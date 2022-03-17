using LinearAlgebra, Flux, Optim
using Zygote: ignore, dropgrad

#Note: we use two Q networks, this is not used in the original publications, but there is no reason to not do it since the networks are trained the same way as for example SAC
mutable struct MPOPolicy{P,Q,R}
    policy::P
    qnetwork1::Q,
    qnetwork2::Q,
    target_qnetwork1::Q
    target_qnetwork2::Q 
    γ::Float32 = 0.99f0
    batch_size::Int #N
    action_sample_size::Int #K 
    ϵ::Float32  #KL bound on the non-parametric variational approximation to the policy
    ϵμ::Float32 #KL bound for the parametric policy training of mean estimations
    ϵΣ::Float32 #KL bound for the parametric policy training of (co)variance estimations
    αμ::Float32
    αΣ::Float32
    update_freq::Int
    update_after::Int
    update_step::Int
    batches_per_update::Int
    τ::Float32 #Polyak avering parameter of target networks
    rng::R
end
end

function (p::MPOPolicy)(env)
    D = device(p.policy)
    s = send_to_device(D, state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    action = dropdims(p.policy(p.rng, s; is_sampling=true), dims=2) # Single action vec, drop second dim
    send_to_host(action)
end

function RLBase.update!(
    p::MPOPolicy,
    traj::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    length(traj) > p.update_after || return
    p.update_step % p.update_freq == 0 || return
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
    update!(p, batch)
end

function RLBase.update!(p::MPOPolicy, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(p.qnetwork1), batch)

    γ, τ, α = p.γ, p.τ, p.α

    a′, log_π = p.policy(p.rng, s′; is_sampling=true, is_return_log_prob=true)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec(q′ .- α .* log_π)

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

    #Fit non-parametric variational distribution
    states = Flux.unsqueeze(s,2) #3D tensor with dimensions (state_size x 1 x batch_size)
    action_samples, logp_π = p.policy(p.rng, states, p.action_sample_size) #3D tensor with dimensions (action_size x action_sample_size x batchsize)
    input = vcat(repeat(states, outer = (1, p.action_sample_size, 1)), action_samples) #repeat states along 2nd dimension and vcat with sampled actions to get state-action tensor
    Q = p.qnetwork1(input)
    η = solve_mpodual(Q, p.ϵ, p.policy)
    qij = softmax(Q./η, dims = 2) # dims = (1 x actions_sample_size x batch_size)


    #Improve policy towards qij
    μ_old, Σ_old = p.policy(p.rng, states, is_sampling = false)
    ps = Flux.params(p.policy, p.αμ, p.αΣ)
    gs = gradient(ps) do 
        loss_decoupled(p, qij, states, actions, μ_old, Σ_old)
    end
    
    if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), gs)
        error("Gradient contains NaN of Inf")
    end

    gs[p.αμ] .*= -1 #negative of gradient since we minimize w.r.t. α
    gs[p.αΣ] .*= -1 

    Flux.Optimise.update!(p.optimizer, ps, gs)
    clamp!(p.αμ, 0f0, Inf32) #maybe add an upperbound ?
    clamp!(p.αΣ, 0f0, Inf32)
end


function solve_mpodual(Q, ϵ, ::GaussianNetwork)
    max_Q = maximum(Q, dims = 1) #needed for numerical stability
    g(η) = only(η .* p.ϵ .+ mean(max_q) .+ η .* mean(log.(mean(exp.((Q .- max_Q)./η),dims = 1)),dims = 2))
    η = only(Optim.minimizer(optimize(g, [eps(ϵ)]))) #uses Nelder-Mead's algorithm, a more efficient way to optimize a convex scalar function than Adam
end

#TODO: Dual for the Discrete case, needs probability vector
function solve_mpodual(Q, ϵ, ::NeuralNetworkApproximator)
    max_Q = maximum(Q, dims = 1) #needed for numerical stability
    g(η) = only(η .* p.ϵ .+ mean(max_q) .+ η .* mean(log.(mean(exp.((Q .- max_Q)./η),dims = 1)),dims = 2))
    only(Optim.minimizer(optimize(g, [eps(ϵ)]))) #uses Nelder-Mead's algorithm, a more efficient way to optimize a convex scalar function than Adam
end

function loss_decoupled(p::MPOPolicy, qij, states, actions, μ_old, σ_old)
    μ, logσ = p.policy(p.rng, states, is_sampling = false) #3D tensors with dimensions (action_size x action_size x batch_size)
    σ = exp.(logσ)
    σ_d = Zygote.ignore(σ) #decoupling
    μ_d = Zygote.ignore(μ)
    #decoupled logp for mu and sigma
    logp_π_new_μ = sum(normlogpdf(μ, σ_d, actions) .- (2.0f0 .* (log(2.0f0) .- actions .- softplus.(-2.0f0 .* actions))), dims = 1)
    logp_π_new_σ = sum(normlogpdf(μ_d, σ, actions) .- (2.0f0 .* (log(2.0f0) .- actions .- softplus.(-2.0f0 .* actions))), dims = 1)
    policy_loss = sum(qij .* (logp_π_new_μ .+ logp_π_new_σ)) 
    lagrangeloss = p.αμ * (p.ϵμ - mean(gaussiankl(μ_old, σ_old, μ, σ_d))) + p.αΣ *(p.ϵΣ - mean(gaussiankl(μ_old, σ_old, μ_d, σ)))
    
    return policy_loss + lagrangeloss
end

#In the case of diagonal covariance (with GaussianNetwork), the diagonals are squeezed to Matrix column vectors 
function loss_decoupled(p::MPOPolicy{GaussianNetwork}, qij, states, actions, μ_old, σ_old)
    μ, logσ = p.policy(p.rng, dropdims(states,dims = 2), is_sampling = false) #3D tensors with dimensions (action_size x 1 x batch_size)
    σ = exp.(logσ)
    σ_d = Zygote.ignore(σ) #decoupling
    μ_d = Zygote.ignore(μ)
    #decoupled logp for mu and sigma
    logp_π_new_μ = sum(normlogpdf(μ, σ_d, actions) .- (2.0f0 .* (log(2.0f0) .- actions .- softplus.(-2.0f0 .* actions))), dims = 1)
    logp_π_new_σ = sum(normlogpdf(μ_d, σ, actions) .- (2.0f0 .* (log(2.0f0) .- actions .- softplus.(-2.0f0 .* actions))), dims = 1)
    policy_loss = sum(qij .* (logp_π_new_μ .+ logp_π_new_σ))
    tomatrix = m -> dropdims(m, dims = 2)
    lagrangeloss =  p.αμ * (p.ϵμ - mean(gaussiankl(tomatrix(μ_old), tomatrix(σ_old), tomatrix(μ), tomatrix(σ_d)))) + 
                    p.αΣ * (p.ϵΣ - mean(gaussiankl(tomatrix(μ_old), tomatrix(σ_old), tomatrix(μ_d), tomatrix(σ))))
    
    return policy_loss + lagrangeloss
end

#computes the KL divergence between two multivariate gaussians with diagonal variances. Parameters must be input as Matrices
function gaussiankl(μ1::AbstractVecOrMat, σ1::AbstractVecOrMat, μ2::AbstractVecOrMat, σ2::AbstractVecOrMat)
    d = size(μ,1)
    square_diff = (μ2 .- μ1) .^2
    1/2 .*(log.(prod(σ2, dims = 1)/prod(σ1, dims = 1)) .- d .+ prod(σ1 ./ σ2, dims = 1) .+ sum(square_diff .* 1 ./ σ2, dims = 1))
end

#gaussian KL with triangular or full covariance, so with 3D Σ inputs mapslices(det, Σ, dims = (1,2))
function gaussiankl(μ1, Σ1::AbstractArray, μ2, Σ2::AbstractArray)
    mapslices3(f, t) = t -> mapslices(f,t, dims = (1,2)) #apply function f to all matrices along the 3rd dimension of tensor t
    d = size(μ,1)
    diffs = μ2 .- μ1 #action_size x 1 x batch_size
    
    logdet = log.(mapslices3(det, Σ2) ./ mapslices3(det, Σ1)) .- d #(1x1xbatchsize)
    Σ2i = mapslices3(inv, Σ2)
    mult = similar(Σ1)
    for i in 1:size(Σ1,3)
        mult[:,:,i] = Σ2i[:,:,i]*Σ1[:,:,i]
    end
    trace = mapslices3(tr, Σ2i*Σ1) #1 x 1 x batchsize
    div = similar(trace)
    for i in 1:size(Σ1,3)
        div[:,:,i] = diffs[:,:,i]'*Σ2i[:,:,i]*diffs[:,:,i]
    end
    1/2 .* (logdet .+ trace .+ div)
end


