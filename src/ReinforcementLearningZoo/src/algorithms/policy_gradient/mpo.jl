export MPOPolicy
using LinearAlgebra, Flux, Optim, StatsBase, Random, CUDA
using Zygote: ignore, dropgrad
import LogExpFunctions.logsumexp
import Flux.Losses: logitcrossentropy, mse
import Flux.onehotbatch

#Note: we use two Q networks, this is not used in the original publications, but there is no reason to not do it since the networks are trained the same way as for example SAC
#If using a CategoricalNetwork actor, α is used for αμ. αΣ is only used for [Cov]GaussianNetwork
mutable struct MPOPolicy{P<:Approximator,Q<:Approximator,R} <: AbstractPolicy
    actor::P
    qnetwork1::Q
    qnetwork2::Q
    target_qnetwork1::Q
    target_qnetwork2::Q 
    γ::Float32
    action_sample_size::Int #K 
    ϵ::Float32  #KL bound on the non-parametric variational approximation to the actor
    ϵμ::Float32 #KL bound for the parametric actor training of mean estimations
    ϵΣ::Float32 #KL bound for the parametric actor training of (co)variance estimations
    α::Float32
    αΣ::Float32
    α_scale::Float32 #learning rate for α
    αΣ_scale::Float32 #learning rate for αΣ
    max_grad_norm::Float32
    τ::Float32 #Polyak avering parameter of target networks
    rng::R
    logs::Dict{Symbol, Vector{Float32}}
end


"""
    MPOPolicy(;
    actor::Approximator, #The policy approximating function. Can be a GaussianNetwork, a CovGaussianNetwork or a CategoricalNetwork.
    qnetwork1::Q <: Approximator, #The Q-Value approximating function.  
    qnetwork2::Q, #A second Q-Value approximator for double Q-learning.
    γ = 0.99f0, #Discount factor of rewards.
    action_sample_size::Int, #The number of actions to sample at the E-step (K in the MPO paper).
    ϵ = 0.1f0, #maximum kl divergence between the current policy and the E-step empirical policy.
    ϵμ = 5f-4, #maximum kl divergence between the current policy and the updated policy at the M-step w.r.t the mean or the logits.
    ϵΣ = 1f-5, #maximum kl divergence between the current policy and the updated policy at the M-step w.r.t the covariance (not used with categorical policy).
    α_scale = 1f0, #gradient descent learning rate for the lagrange penalty.
    αΣ_scale = 100f0, #gradient descent learning rate for the lagrange penalty for the covariance decoupling (not used with categorical policy).
    τ = 1f-3, #polyak-averaging update parameter for the target Q-networks.
    max_grad_norm = 5f-1, #maximum gradient norm.
    rng = Random.GLOBAL_RNG
    )

Instantiate an MPO learner. The actor can be of type `GaussianNetwork`, `CovGaussianNetwork`,
or `CategoricalNetwork`. The original paper uses `CovGaussianNetwork` which approximates a 
MvNormal policy. It has a better policy representation but requires more computation.

This implementation uses double Q-learning and target Q-networks, 
unlike the original MPO paper that uses retrace (WIP). The `Approximator` fields should 
each come with their own `Optimiser` (e.g. `ADAM`). 

MPOPolicy requires batches of type 
```
NamedTuple{
        (:actor, :critic), 
        <: Tuple{
            <: Vector{<: NamedTuple{(:state,)}},
            <: Vector{<: NamedTuple{SS′ART}}
        }
    }
``` 
to be trained. This type is obtained with a MetaSampler with two MultiBatchSampler: 
a `:actor` and a `:critic` one. The :actor sampler must sample :state traces only 
and the :critic needs SS′ART traces. See `ReinforcementLearningExperiments` for examples
with each policy network type.

`p::MPOPolicy` logs several values during training. You can access them using `p.logs[::Symbol]`.
"""
function MPOPolicy(;actor::Approximator, qnetwork1::Q, qnetwork2::Q, γ = 0.99f0, action_sample_size::Int, ϵ = 0.1f0, ϵμ = 5f-4, ϵΣ = 1f-5, α_scale = 1f0, αΣ_scale = 100f0, τ = 1f-3, max_grad_norm = 5f-1, rng = Random.GLOBAL_RNG) where Q <: Approximator
    @assert device(actor) == device(qnetwork1) == device(qnetwork2) "All network approximators must be on the same device"
    @assert device(actor) == device(rng) "The specified rng does not generate on the same device as the actor. Use `CUDA.CURAND.RNG()` to work with a CUDA GPU"
    logs = Dict(s => Float32[] for s in (:qnetwork1_loss, :qnetwork2_loss, :actor_loss, :lagrangeμ_loss, :lagrangeΣ_loss, :η, :α, :αΣ, :kl))
    MPOPolicy(actor, qnetwork1, qnetwork2, deepcopy(qnetwork1), deepcopy(qnetwork2), γ, action_sample_size, ϵ, ϵμ, ϵΣ, 0f0, 0f0, α_scale, αΣ_scale, max_grad_norm, τ, rng, logs)
end

Flux.@functor MPOPolicy

function (p::MPOPolicy)(env; testmode = false)
    D = device(p.actor)
    s = send_to_device(D, state(env))
    if !testmode
        action = p.actor.model(p.rng, s; is_sampling=!testmode)
    else
        action, _ = p.actor.model(p.rng, s; is_sampling=!testmode)
    end
    send_to_host(action)
end

function RLBase.optimise!(
    p::MPOPolicy,
    batches::NamedTuple{
        (:actor, :critic), 
        <: Tuple{
            <: Vector{<: NamedTuple{(:state,)}},
            <: Vector{<: NamedTuple{SS′ART}}
        }
    }
)
    update_critic!(p, batches[:critic])
    update_actor!(p, batches[:actor])
end

#Here we apply the TD3 Q network approach. The original MPO paper uses retrace.
function update_critic!(p::MPOPolicy, batches)
    modulo = rand(p.rng, (0,1)) #we randomize this so that if the number of batches is odd, we do not train one critic more than the other.
    for (id, batch) in enumerate(batches)
        s, s′, a, r, t, = send_to_device(device(p.qnetwork1), batch)
        γ, τ = p.γ, p.τ

        a′ = p.actor(p.rng, s′; is_sampling=true, is_return_log_prob=false)
        q′_input = vcat(s′, a′)
        q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

        y =  r .+ γ .* (1 .- t) .* vec(q′) 

        # Train Q Networks
        q_input = vcat(s, a)
        if id % 2 == modulo
            q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
                q1 = p.qnetwork1(q_input) |> vec
                l = mse(q1, y)
                ignore() do 
                    push!(p.logs[:qnetwork1_loss], l)
                end
                return l
            end
            if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), q_grad_1)
                error("Gradient of Q_1 contains NaN of Inf")
            end
            Flux.Optimise.update!(p.qnetwork1.optimiser, Flux.params(p.qnetwork1), q_grad_1)
        else
            q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
                q2 = p.qnetwork2(q_input) |> vec
                l = mse(q2, y)
                ignore() do 
                    push!(p.logs[:qnetwork2_loss], l)
                end
                return l
            end
            if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), q_grad_2)
                error("Gradient of Q_2 contains NaN of Inf")
            end
            Flux.Optimise.update!(p.qnetwork2.optimiser, Flux.params(p.qnetwork2), q_grad_2)
        end

        for (dest, src) in zip(
            Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
            Flux.params([p.qnetwork1, p.qnetwork2]),
        )
            dest .= (1 - τ) .* dest .+ τ .* src
        end
    end
end

function update_actor!(p::MPOPolicy, batches::Vector{<:NamedTuple{(:state,)}})
    states_batches = [send_to_device(device(p.actor), reshape(batch[:state], size(batch[:state],1), 1, :)) for batch in batches] #vector of 3D tensors with dimensions (state_size x 1 x batch_size), sent to device
    current_action_dist_batches = [p.actor(p.rng, states, is_sampling = false) for states in states_batches] #π(.|s,Θᵢ) 
    action_samples_batches = [sample_actions(p, dist, p.action_sample_size) for dist in current_action_dist_batches] #3D tensor with dimensions (action_size x action_sample_size x batchsize)
    for (states, current_action_dist, action_samples) in zip(states_batches, current_action_dist_batches, action_samples_batches)
        #Fit non-parametric variational distributions (E-step)
        repeated_states = reduce(hcat, Iterators.repeated(states, p.action_sample_size))
        input = vcat(repeated_states, action_samples) #repeat states along 2nd dimension and vcat with sampled actions to get state-action tensor
        Q = p.qnetwork1(input) 
        η = solve_mpodual(send_to_host(Q), p.ϵ)
        push!(p.logs[:η], η)
        qij = softmax(Q./η, dims = 2) # dims = (1 x actions_sample_size x batch_size)

        if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), qij)
            error("qij contains NaN of Inf")
        end

        #Improve actor towards qij (M-step)
        ps = Flux.params(p.actor)#, p.α, p.αΣ)
        gs = gradient(ps) do 
            mpo_loss(p, qij, states, action_samples, current_action_dist)
        end
        
        if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), gs)
            error("Gradient contains NaN of Inf")
        end

        grad_norm!(gs, p.max_grad_norm)

        Flux.Optimise.update!(p.actor.optimiser, ps, gs)

        ignore() do 
            push!(p.logs[:α], p.α)
            push!(p.logs[:αΣ],p.αΣ)
        end
    end
end

function sample_actions(p::MPOPolicy{<:Approximator{<:CovGaussianNetwork}}, dist, N::Int)
    μ, L = dist
    noise = randn(p.rng, eltype(μ), size(μ,1), N, size(μ,3))
    output = similar(noise)
    for k in axes(μ,3)
        output[:,:,k] .= μ[:,:,k] .+ L[:,:,k] .* noise[:,:,k]
    end
    output
end

function sample_actions(p::MPOPolicy{<:Approximator{<:GaussianNetwork}}, dist, N::Int)
    μ, σ = dist
    noise = randn(p.rng, eltype(μ), size(μ,1), N, size(μ,3))
    μ .+ σ .* noise
end

function sample_actions(p::MPOPolicy{<:Approximator{<:CategoricalNetwork}}, logits, N)
    batch_size = size(logits, 3) #3
    da = size(logits, 1)
    log_probs = logsoftmax(logits, dims = 1)
    gumbels = -log.(-log.(rand(p.rng, da, N, batch_size))) .+ log_probs # Gumbel-Max trick
    z = getindex.(argmax(gumbels, dims = 1), 1)
    reshape(onehotbatch(z, 1:size(logits,1)), size(gumbels)...) # reshape to 3D due to onehotbatch behavior
end

function solve_mpodual(Q::AbstractArray, ϵ)    
    g(η) = η * ϵ + η * mean(logsumexp( Q ./η .- Float32(log(size(Q, 2))), dims = 2))
    Optim.minimizer(optimize(g, eps(ϵ), 10f0))
end

#For CovGaussianNetwork
function mpo_loss(p::MPOPolicy{<:Approximator{<:CovGaussianNetwork}}, qij, states, actions, μ_L_old::Tuple)
    μ_old, L_old = μ_L_old
    μ, L = p.actor(p.rng, states, is_sampling = false)
    #decoupling
    μ_d, L_d = ignore() do 
        μ, L 
    end 
    #decoupled logp for mu and L
    logp_π_new_μ = mvnormlogpdf(μ, L_d, actions) 
    logp_π_new_L = mvnormlogpdf(μ_d, L, actions)
    actor_loss = - mean(qij .* (logp_π_new_μ .+ logp_π_new_L))
    μ_old_s, L_old_s, μ_s, L_d_s, μ_d_s, L_s = map(x->eachslice(x, dims =3), (μ_old, L_old, μ, L_d, μ_d, L)) #slice all tensors along 3rd dim

    klμ = mean(mvnormkldivergence.(μ_old_s, L_old_s, μ_s, L_d_s))
    klΣ = mean(mvnormkldivergence.(μ_old_s, L_old_s, μ_d_s, L_s))
    
    ignore() do
        p.α -= p.α_scale*(p.ϵμ - klμ) 
        p.αΣ -= p.αΣ_scale*(p.ϵμ - klΣ) 
        p.α = clamp(p.α, 0f0, Inf32)
        p.αΣ = clamp(p.αΣ, 0f0, Inf32)
    end

    lagrangeμ = - p.α * (p.ϵμ - klμ) 
    lagrangeΣ = - p.αΣ * (p.ϵΣ - klΣ)
    ignore() do #logging
        push!(p.logs[:actor_loss],actor_loss)
        push!(p.logs[:lagrangeμ_loss], lagrangeμ)
        push!(p.logs[:lagrangeΣ_loss], lagrangeΣ)
        push!(p.logs[:kl], klμ)
    end
    return actor_loss + lagrangeμ + lagrangeΣ
end

#In the case of diagonal covariance (with GaussianNetwork), 
function mpo_loss(p::MPOPolicy{<:Approximator{<:GaussianNetwork}}, qij, states, actions, μ_logσ_old::Tuple)
    μ_old, logσ_old = μ_logσ_old
    σ_old = exp.(logσ_old)
    μ, logσ = p.actor(p.rng, states, is_sampling = false) #3D tensors with dimensions (action_size x 1 x batch_size)
    σ = exp.(logσ)
    μ_d, σ_d = ignore() do
        μ, σ #decoupling
    end
    #decoupled logp for mu and sigma
    μ_old_s, σ_old_s, μ_s, σ_d_s, μ_d_s, σ_s = map(x->eachslice(x, dims =3), (μ_old, σ_old, μ, σ_d, μ_d, σ)) #slice all tensors along 3rd dim

    logp_π_new_μ = diagnormlogpdf(μ, σ_d, actions)
    logp_π_new_σ = diagnormlogpdf(μ_d, σ, actions)
    actor_loss = -mean(qij .* (logp_π_new_μ .+ logp_π_new_σ))
    klμ = mean(diagnormkldivergence.(μ_old_s, σ_old_s, μ_s, σ_d_s))
    klΣ = mean(diagnormkldivergence.(μ_old_s, σ_old_s, μ_d_s, σ_s))
    
    ignore() do
        p.α -= p.α_scale*(p.ϵμ - klμ) 
        p.αΣ -= p.αΣ_scale*(p.ϵμ - klΣ) 
        p.α = clamp(p.α, 0f0, Inf32)
        p.αΣ = clamp(p.αΣ, 0f0, Inf32)
    end

    lagrangeμ = - p.α * (p.ϵμ - klμ) 
    lagrangeΣ = - p.αΣ * (p.ϵΣ - klΣ)
    ignore() do #logging
        push!(p.logs[:actor_loss],actor_loss)
        push!(p.logs[:lagrangeμ_loss], lagrangeμ)
        push!(p.logs[:lagrangeΣ_loss], lagrangeΣ)
        push!(p.logs[:kl], klμ)
    end
    return actor_loss + lagrangeμ + lagrangeΣ
end

function mpo_loss(p::MPOPolicy{<:Approximator{<:CategoricalNetwork}}, qij, states, actions, logits_old)
    logits = p.actor(p.rng, states, is_sampling = false) #3D tensors with dimensions (action_size x 1 x batch_size)
    actor_loss = -  mean(qij .* log.(sum(softmax(logits, dims = 1) .* actions, dims = 1)))
    kl = kldivergence(softmax(logits_old, dims = 1), softmax(logits, dims = 1))/prod(size(qij)[2:3]) #divide to get average
    
    ignore() do 
        p.α -= 1*(p.ϵμ - kl)
        p.α = clamp(p.α, 0f0, Inf32)
    end

    lagrange_loss = - p.α * (p.ϵμ - kl)
    ignore() do #logging
        push!(p.logs[:actor_loss],actor_loss)
        push!(p.logs[:lagrangeμ_loss], lagrange_loss)
        push!(p.logs[:kl], kl)
    end
    return actor_loss + lagrange_loss
end

"""
    Flux.gpu(p::MPOPolicy; rng = CUDA.CURAND.RNG())

Apply Flux.gpu to all neural nets of the actor. 
`rng` can be used to specificy a particular rng if desired, make sure this rng generates numbers on the correct device.
"""
function Flux.gpu(p::MPOPolicy; rng = CUDA.CURAND.RNG())
    MPOPolicy(
        p.actor |> gpu, 
        p.qnetwork1 |> gpu, 
        p.qnetwork2 |> gpu, 
        p.target_qnetwork1 |> gpu, 
        p.target_qnetwork2 |> gpu, 
        p.γ, 
        p.action_sample_size, 
        p.ϵ, 
        p.ϵμ, 
        p.ϵΣ, 
        p.α, 
        p.αΣ,
        p.α_scale,
        p.αΣ_scale,
        p.max_grad_norm,
        p.τ, 
        rng,
        p.logs)
end

#=
    send_to_device(device, p::MPOPolicy; rng = device isa CuDevice ? CUDA.CURAND.RNG() : GLOBAL_RNG)

Send all neural nets of the actor to a specified device.
`rng` can be used to specificy a particular rng if desired, make sure this rng generates numbers on `device`. 
=#
function RLCore.send_to_device(device, p::MPOPolicy; rng = device isa CuDevice ? CUDA.CURAND.RNG() : GLOBAL_RNG)
    sd(x) = send_to_device(device, x) 
    MPOPolicy(
        p.actor |> sd, 
        p.qnetwork1 |> sd, 
        p.qnetwork2 |> sd, 
        p.target_qnetwork1 |> sd, 
        p.target_qnetwork2 |> sd, 
        p.γ,
        p.action_sample_size, 
        p.ϵ, 
        p.ϵμ, 
        p.ϵΣ, 
        p.α, 
        p.αΣ, 
        p.α_scale,
        p.αΣ_scale,
        p.max_grad_norm,
        p.τ, 
        rng,
        p.logs)
end

#=
    send_to_host(p::MPOPolicy; rng = GLOBAL_RNG)

Send all neural nets of the actor to the cpu.
`rng` can be used to specificy a particular rng if desired. 
=#
function RLCore.send_to_host(p::MPOPolicy; rng = GLOBAL_RNG)
    send_to_device(Val{:cpu}, p, rng = rng)
end

function grad_norm!(grad, max_norm)
    n = norm(grad)
    if n > max_norm
        for g in grad
            rmul!(g, max_norm/n)
        end
    end
    return grad
end