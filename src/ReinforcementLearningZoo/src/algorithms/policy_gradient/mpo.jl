export MPOPolicy
using LinearAlgebra, Flux, Optim
using Zygote: ignore, dropgrad
import LogExpFunctions.logsumexp
import Flux.Losses: logitcrossentropy, mse

#Note: we use two Q networks, this is not used in the original publications, but there is no reason to not do it since the networks are trained the same way as for example SAC
mutable struct MPOPolicy{P<:Approximator,Q<:Approximator,R,AV<:AbstractVector, N} <: AbstractPolicy
    policy::P
    qnetwork1::Q
    qnetwork2::Q
    target_qnetwork1::Q
    target_qnetwork2::Q 
    γ::Float32
    action_sample_size::Int #K 
    ϵ::Float32  #KL bound on the non-parametric variational approximation to the policy
    ϵμ::Float32 #KL bound for the parametric policy training of mean estimations
    ϵΣ::Float32 #KL bound for the parametric policy training of (co)variance estimations
    αμ::AV
    αΣ::AV
    τ::Float32 #Polyak avering parameter of target networks
    rng::R
    logs::Dict{Symbol, Vector{Float32}}
end

function MPOPolicy(;policy::Approximator, qnetwork1::Q, qnetwork2::Q, γ = 0.99f0, action_sample_size, ϵ = 0.1f0, ϵμ = 5f-4, ϵΣ = 1f-5, τ = 1f-3, rng = Random.GLOBAL_RNG) where Q <: Approximator
    @assert device(policy) == device(qnetwork1) == device(qnetwork2) "All network approximators must be on the same device"
    @assert device(policy) == device(rng) "The specified rng does not generate on the same device as the policy. Use `CUDA.CURAND.RNG()` to work with a CUDA GPU"
    αμ = send_to_device(device(policy), [0f0])
    αΣ = send_to_device(device(policy), [0f0])
    logs = Dict(s => Float32[] for s in (:qnetwork1_loss, :qnetwork2_loss, :policy_loss, :lagrangeμ_loss, :lagrangeΣ_loss, :η, :αμ, :αΣ, :klμ, :klΣ))
    MPOPolicy(policy, qnetwork1, qnetwork2, deepcopy(qnetwork1), deepcopy(qnetwork2), γ, action_sample_size, ϵ, ϵμ, ϵΣ, αμ, αΣ, τ, rng, logs)
end

Flux.@functor MPOPolicy

function (p::MPOPolicy)(env; testmode = false)
    p.update_step += 1
    D = device(p.policy)
    s = send_to_device(D, state(env))
    if !testmode
        action = p.policy(p.rng, s; is_sampling=!testmode)
    else
        action, _ = p.policy(p.rng, s; is_sampling=!testmode)
    end
    send_to_host(action)
end

#=
```
NamedTuple{
        (:policy, :critic), 
        <: Tuple{
            <: Vector{<: NamedTuple{(:state)}},
            <: Vector{<: NamedTuple{SS′ART}}
        }
    }
```
is the type of batches returned by a MetaSampler with two MutliBatchSampler: a `:policy` and a `:critic` one.
The :policy sampler must sample :state traces only and the :critic needs SS′ART traces. 

=#
function RLBase.optimise!(
    p::MPOPolicy,
    batches::NamedTuple{
        (:policy, :critic), 
        <: Tuple{
            <: Vector{<: NamedTuple{(:state)}},
            <: Vector{<: NamedTuple{SS′ART}}
        }
    }
)
    for batch in batches[:critic]
        update_critic!(p, batch)
    end
    for batch in batches[:policy]
        update_policy!(p, batch)
    end
end

#Here we apply the TD3 Q network approach. This could be customizable by the user in a new p.critic <: AbstractCritic field. 
function update_critic!(p::MPOPolicy, batch)
    s, a, r, t, s′ = send_to_device(device(p.qnetwork1), batch)
    γ, τ = p.γ, p.τ

    a′ = p.policy(p.rng, s′; is_sampling=true, is_return_log_prob=false)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y =  r .+ γ .* (1 .- t) .* vec(q′) 

    # Train Q Networks
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
        q1 = p.qnetwork1(q_input) |> vec
        l = mse(q1, y)
        ignore_derivatives() do 
            push!(p.logs[:qnetwork1_loss], l)
        end
        return l
    end
    if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), q_grad_1)
        error("Gradient of Q_1 contains NaN of Inf")
    end
    update!(p.qnetwork1, q_grad_1)
    q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
        q2 = p.qnetwork2(q_input) |> vec
        l = mse(q2, y)
        ignore_derivatives() do 
            push!(p.logs[:qnetwork2_loss], l)
        end
        return l
    end
    if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), q_grad_2)
        error("Gradient of Q_2 contains NaN of Inf")
    end
    update!(p.qnetwork2, q_grad_2)

    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end
end

function update_policy!(p::MPOPolicy, batch::NamedTuple{(:state)})
    states = send_to_device(device(p.policy), reshape(batch[:state], size(batch[:state],1), 1, :)) #3D tensors with dimensions (state_size x 1 x batch_size), sent to device
    current_action_dist = p.policy(p.rng, states, is_sampling = false)

    #Fit non-parametric variational distribution
    action_samples = p.policy(p.rng, states, p.action_sample_size, is_return_log_prob = false) #3D tensor with dimensions (action_size x action_sample_size x batchsize)
    repeated_states = reduce(hcat, Iterators.repeated(states, p.action_sample_size))
    input = vcat(repeated_states, action_samples) #repeat states along 2nd dimension and vcat with sampled actions to get state-action tensor
    Q = p.qnetwork1(input) 
    η = solve_mpodual(send_to_host(Q), p.ϵ)
    push!(p.logs[:η], η)
    qij = softmax(Q./η, dims = 2) # dims = (1 x actions_sample_size x batch_size)

    if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), qij)
        error("qij contains NaN of Inf")
    end

    #Improve policy towards qij
    ps = Flux.params(p.policy, p.αμ, p.αΣ)
    gs = gradient(ps) do 
        mpo_loss(p, qij, states, action_samples, current_action_dist)
    end
    
    if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), gs)
        error("Gradient contains NaN of Inf")
    end

    gs[p.αμ] *= -1 #negative of gradient since we maximize w.r.t. α
    gs[p.αΣ] *= -1 

    Flux.Optimise.update!(p.policy.optimizer, ps, gs)
    p.αμ = clamp.(p.αμ, 0f0, Inf32) #maybe add an upperbound ?
    p.αΣ = clamp.(p.αΣ, 0f0, Inf32)
    ignore_derivatives() do 
        push!(p.logs[:αμ],sum(p.αμ))
        push!(p.logs[:αΣ],sum(p.αΣ))
    end
end

function solve_mpodual(Q::AbstractArray, ϵ)    
    g(η) = η * ϵ + η * mean(logsumexp( Q ./η .- log(size(Q, 2)*1f0), dims = 2))
    Optim.minimizer(optimize(g, eps(ϵ), 10f0))
end

#For CovGaussianNetwork
function mpo_loss(p::MPOPolicy{<:Approximator{<:CovGaussianNetwork}}, qij, states, actions, μ_L_old::Tuple)
    μ_old, L_old = μ_L_old
    μ, L = p.policy(p.rng, states, is_sampling = false)
    #decoupling
    μ_d, L_d = ignore_derivatives() do 
        μ, L 
    end 
    #decoupled logp for mu and L
    logp_π_new_μ = mvnormlogpdf(μ, L_d, actions) 
    logp_π_new_L = mvnormlogpdf(μ_d, L, actions)
    policy_loss = - mean(qij .* (logp_π_new_μ .+ logp_π_new_L))
    μ_old_s, L_old_s, μ_s, L_d_s, μ_d_s, L_s = map(x->eachslice(x, dims =3), (μ_old, L_old, μ, L_d, μ_d, L)) #slice all tensors along 3rd dim

    klμ = mean(mvnormkldivergence.(μ_old_s, L_old_s, μ_s, L_d_s))
    klΣ = mean(mvnormkldivergence.(μ_old_s, L_old_s, μ_d_s, L_s))
    lagrangeμ = - mean(p.αμ) * (p.ϵμ - klμ) 
    lagrangeΣ = - mean(p.αΣ) * (p.ϵΣ - klΣ)
    ignore_derivatives() do #logging
        push!(p.logs[:policy_loss],policy_loss)
        push!(p.logs[:lagrangeμ_loss], lagrangeμ)
        push!(p.logs[:lagrangeΣ_loss], lagrangeΣ)
        push!(p.logs[:klμ], klμ)
        push!(p.logs[:klΣ], klΣ)
    end
    return policy_loss + lagrangeμ + lagrangeΣ
end

#In the case of diagonal covariance (with GaussianNetwork), 
function mpo_loss(p::MPOPolicy{<:Approximator{<:GaussianNetwork}}, qij, states, actions, μ_logσ_old::Tuple)
    μ_old, logσ_old = μ_logσ_old
    σ_old = exp.(logσ_old)
    μ, logσ = p.policy(p.rng, states, is_sampling = false) #3D tensors with dimensions (action_size x 1 x batch_size)
    σ = exp.(logσ)
    μ_d, σ_d = ignore_derivatives() do
        μ, σ #decoupling
    end
    #decoupled logp for mu and sigma
    μ_old_s, σ_old_s, μ_s, σ_d_s, μ_d_s, σ_s = map(x->eachslice(x, dims =3), (μ_old, σ_old, μ, σ_d, μ_d, σ)) #slice all tensors along 3rd dim

    logp_π_new_μ = diagnormlogpdf(μ, σ_d, actions)
    logp_π_new_σ = diagnormlogpdf(μ_d, σ, actions)
    policy_loss = -mean(qij .* (logp_π_new_μ .+ logp_π_new_σ))

    lagrangeμ = - mean(p.αμ) * (p.ϵμ - mean(diagnormkldivergence.(μ_old_s, σ_old_s, μ_s, σ_d_s))) 
    lagrangeΣ = - mean(p.αΣ) * (p.ϵΣ - mean(diagnormkldivergence.(μ_old_s, σ_old_s, μ_d_s, σ_s)))
    ignore_derivatives() do #logging
        push!(p.logs[:policy_loss],policy_loss)
        push!(p.logs[:lagrangeμ_loss], lagrangeμ)
        push!(p.logs[:lagrangeΣ_loss], lagrangeΣ)
    end
    return policy_loss + lagrangeμ + lagrangeΣ
end

function mpo_loss(p::MPOPolicy{<:Approximator{<:CategoricalNetwork}}, qij, states, actions, logits_old)
    logits = p.policy(p.rng, states, is_sampling = false) #3D tensors with dimensions (action_size x 1 x batch_size)
    policy_loss = -  mean(qij .* logsoftmax(logits, dims = 2))
    lagrange_loss = - mean(p.αμ) * (p.ϵμ - kldivergence(softmax(logits_old, dims = 1), softmax(logits, dims = 1)))
    ignore_derivatives() do #logging
        push!(p.logs[:policy_loss],policy_loss)
        push!(p.logs[:lagrangeμ_loss], lagrange_loss)
    end
    return policy_loss + lagrange_loss
end

"""
    Flux.gpu(p::MPOPolicy; rng = CUDA.CURAND.RNG())

Apply Flux.gpu to all neural nets of the policy. 
`rng` can be used to specificy a particular rng if desired, make sure this rng generates numbers on the correct device.
"""
function Flux.gpu(p::MPOPolicy; rng = CUDA.CURAND.RNG())
    MPOPolicy(
        p.policy |> gpu, 
        p.qnetwork1 |> gpu, 
        p.qnetwork2 |> gpu, 
        p.target_qnetwork1 |> gpu, 
        p.target_qnetwork2 |> gpu, 
        p.γ, 
        p.action_sample_size, 
        p.ϵ, 
        p.ϵμ, 
        p.ϵΣ, 
        p.αμ |> gpu, 
        p.αΣ |> gpu,
        p.τ, 
        rng,
        p.logs)
end

#=
    send_to_device(device, p::MPOPolicy; rng = device isa CuDevice ? CUDA.CURAND.RNG() : GLOBAL_RNG)

Send all neural nets of the policy to a specified device.
`rng` can be used to specificy a particular rng if desired, make sure this rng generates numbers on `device`. 
=#
function RLCore.send_to_device(device, p::MPOPolicy; rng = device isa CuDevice ? CUDA.CURAND.RNG() : GLOBAL_RNG)
    sd(x) = send_to_device(device, x) 
    MPOPolicy(
        p.policy |> sd, 
        p.qnetwork1 |> sd, 
        p.qnetwork2 |> sd, 
        p.target_qnetwork1 |> sd, 
        p.target_qnetwork2 |> sd, 
        p.γ,
        p.action_sample_size, 
        p.ϵ, 
        p.ϵμ, 
        p.ϵΣ, 
        p.αμ |> sd, 
        p.αΣ |> sd, 
        p.τ, 
        rng,
        p.logs)
end

#=
    send_to_host(p::MPOPolicy; rng = GLOBAL_RNG)

Send all neural nets of the policy to the cpu.
`rng` can be used to specificy a particular rng if desired. 
=#
function RLCore.send_to_host(p::MPOPolicy; rng = GLOBAL_RNG)
    send_to_device(Val{:cpu}, p, rng = rng)
end
