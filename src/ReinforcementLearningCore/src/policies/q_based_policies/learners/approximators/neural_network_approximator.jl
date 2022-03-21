export NeuralNetworkApproximator, ActorCritic, GaussianNetwork, CovGaussianNetwork, DuelingNetwork, PerturbationNetwork
export VAE, decode, vae_loss

using LinearAlgebra
using Flux
using Random
using Distributions: Normal, logpdf
import Functors: functor
using MacroTools: @forward

"""
    NeuralNetworkApproximator(;kwargs)

Use a DNN model for value estimation.

# Keyword arguments

- `model`, a Flux based DNN model.
- `optimizer=nothing`
"""
Base.@kwdef struct NeuralNetworkApproximator{M,O} <: AbstractApproximator
    model::M
    optimizer::O = nothing
end

# some model may accept multiple inputs
(app::NeuralNetworkApproximator)(args...; kwargs...) = app.model(args...; kwargs...)

@forward NeuralNetworkApproximator.model Flux.testmode!,
Flux.trainmode!,
Flux.params,
device

functor(x::NeuralNetworkApproximator) =
    (model=x.model,), y -> NeuralNetworkApproximator(y.model, x.optimizer)

RLBase.update!(app::NeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, params(app), gs)

Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, params(src))

#####
# ActorCritic
#####

"""
    ActorCritic(;actor, critic, optimizer=ADAM())

The `actor` part must return logits (*Do not use softmax in the last layer!*), and the `critic` part must return a state value.
"""
Base.@kwdef struct ActorCritic{A,C,O} <: AbstractApproximator
    actor::A
    critic::C
    optimizer::O = ADAM()
end

functor(x::ActorCritic) =
    (actor=x.actor, critic=x.critic), y -> ActorCritic(y.actor, y.critic, x.optimizer)

RLBase.update!(app::ActorCritic, gs) = Flux.Optimise.update!(app.optimizer, params(app), gs)

function Base.copyto!(dest::ActorCritic, src::ActorCritic)
    Flux.loadparams!(dest.actor, params(src.actor))
    Flux.loadparams!(dest.critic, params(src.critic))
end

#####
# GaussianNetwork
#####

"""
    GaussianNetwork(;pre=identity, μ, logσ, min_σ=0f0, max_σ=Inf32, normalizer = tanh)

Returns `μ` and `logσ` when called.  Create a distribution to sample from using
`Normal.(μ, exp.(logσ))`. `min_σ` and `max_σ` are used to clip the output from
`logσ`. Actions are normalized according to the specified normalizer function.
"""
Base.@kwdef struct GaussianNetwork{P,U,S,F}
    pre::P = identity
    μ::U
    logσ::S
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
    normalizer::F = tanh
end

GaussianNetwork(pre, μ, logσ, normalizer=tanh) = GaussianNetwork(pre, μ, logσ, 0.0f0, Inf32, normalizer)

Flux.@functor GaussianNetwork

"""
This function is compatible with a multidimensional action space. When outputting an action, it uses the `normalizer` function to normalize it elementwise.

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal distribution. 
- `is_return_log_prob::Bool=false`, whether to calculate the conditional probability of getting actions in the given state.
"""
function (model::GaussianNetwork)(rng::AbstractRNG, s; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    x = model.pre(s)
    μ, raw_logσ = model.μ(x), model.logσ(x)
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))
    if is_sampling
        σ = exp.(logσ)
        z = Zygote.ignore() do
            noise = randn(rng, Float32, size(μ))
            model.normalizer.(μ .+ σ .* noise)
        end
        if is_return_log_prob
            logp_π = sum(normlogpdf(μ, σ, z) .- (2.0f0 .* (log(2.0f0) .- z .- softplus.(-2.0f0 .* z))), dims=1)
            return z, logp_π
        else
            return z
        end
    else
        return μ, logσ
    end
end

"""
    (model::GaussianNetwork)(rng::AbstractRNG, state, action_samples::Int)
Sample `action_samples` actions from each state. Returns a 3D tensor with dimensions (action_size x action_samples x batch_size).
`state` must be 3D tensor with dimensions (state_size x 1 x batch_size). Always returns the logpdf of each action along.
"""
function (model::GaussianNetwork)(rng::AbstractRNG, s, action_samples::Int)
    x = model.pre(s)
    μ, raw_logσ = model.μ(x), model.logσ(x)
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))

    σ = exp.(logσ)
    z = Zygote.ignore() do
        noise = randn(rng, Float32, (size(μ, 1), action_samples, size(μ, 3))...)
        model.normalizer.(μ .+ σ .* noise)
    end
    logp_π = sum(normlogpdf(μ, σ, z) .- (2.0f0 .* (log(2.0f0) .- z .- softplus.(-2.0f0 .* z))), dims=1)
    return z, logp_π
end

function (model::GaussianNetwork)(state; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    model(Random.GLOBAL_RNG, state; is_sampling=is_sampling, is_return_log_prob=is_return_log_prob)
end

function (model::GaussianNetwork)(state, action_samples::Int)
    model(Random.GLOBAL_RNG, state, action_samples)
end

function (model::GaussianNetwork)(state, action)
    x = model.pre(state)
    μ, raw_logσ = model.μ(x), model.logσ(x)
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))
    σ = exp.(logσ)
    logp_π = sum(normlogpdf(μ, σ, action) .- (2.0f0 .* (log(2.0f0) .- action .- softplus.(-2.0f0 .* action))), dims=1)
    return logp_π
end

"""
    CovGaussianNetwork(;pre=identity, μ, Σ, normalizer = tanh)

Returns `μ` and `Σ` when called where μ is the mean and Σ is a covariance matrix. Unlike GaussianNetwork, the output is 3-dimensional.
μ has dimensions (action_size x 1 x batch_size) and Σ has dimensions (action_size x action_size x batch_size).
The Σ head of the `CovGaussianNetwork` should not directly return a square matrix but a vector of length `action_size x (action_size + 1) ÷ 2`. 
This vector will contain elements of the uppertriangular cholesky decomposition of the covariance matrix, which is then reconstructed from it.
Sample from `MvNormal.(μ, Σ)`. Actions are normalized elementwise according to the specified normalizer function.
"""
mutable struct CovGaussianNetwork{P,U,S,F}
    pre::P
    μ::U
    Σ::S
    normalizer::F
end

CovGaussianNetwork(pre, m, s) = CovGaussianNetwork(pre, m, s, tanh)

Flux.@functor CovGaussianNetwork

"""
    (model::CovGaussianNetwork)(rng::AbstractRNG, state; is_sampling::Bool=false, is_return_log_prob::Bool=false)

This function is compatible with a multidimensional action space. When outputting a sampled action, it uses the `normalizer` function to normalize it elementwise.
To work with covariance matrices, the outputs are 3D tensors. 
If sampling, return an actions tensor with dimensions (action_size x action_samples x batch_size) and logp_π (1 x action_samples x batch_size)
If not, returns μ with dimensions (action_size x 1 x batch_size) and L, the lower triangular of the cholesky decomposition of the covariance matrix, with dimensions (action_size x action_size x batch_size)
The covariance matrices can be retrieved with `Σ = Flux.stack(map(l -> l*l', eachslice(L, dims=3)),3)`

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal distribution. 
- `is_return_log_prob::Bool=false`, whether to calculate the conditional probability of getting actions in the given state.
"""
function (model::CovGaussianNetwork)(rng::AbstractRNG, state; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    batch_size = size(state, 3)
    x = model.pre(state)
    μ, cholesky_vec = model.μ(x), model.Σ(x)
    da = size(μ, 1)
    L = vec_to_tril(cholesky_vec, da)

    if is_sampling
        z = Zygote.ignore() do
            noise = randn(rng, eltype(μ), da, 1, batch_size)
            model.normalizer.(Flux.stack(map(.+, eachslice(μ, dims=3), eachslice(L, dims=3) .* eachslice(noise, dims=3)), 3))
        end
        if is_return_log_prob
            logp_π = mvnormlogpdf(μ, L, z)
            return z, logp_π
        else
            return z
        end
    else
        return μ, L
    end
end

"""
    (model::CovGaussianNetwork)(rng::AbstractRNG, state::AbstractMatrix; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    
Given a Matrix of states, will return actions, μ and logpdf in matrix format. The batch of Σ remains a 3D tensor.
"""
function (model::CovGaussianNetwork)(rng::AbstractRNG, state::AbstractMatrix; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    output = model(rng, Flux.unsqueeze(state, 2); is_sampling=is_sampling, is_return_log_prob=is_return_log_prob)
    if output isa Tuple && is_sampling
        dropdims(output[1], dims=2), dropdims(output[2], dims=2)
    elseif output isa Tuple
        dropdims(output[1], dims=2), output[2] #can't reduce the dims of the covariance tensor
    else
        dropdims(output, dims=2)
    end
end



"""
    (model::CovGaussianNetwork)(rng::AbstractRNG, state, action_samples::Int)

Sample `action_samples` actions given `state` and return the `actions, logpdf(actions)`.
This function is compatible with a multidimensional action space. When outputting a sampled action, it uses the `normalizer` function to normalize it elementwise.
The outputs are 3D tensors with dimensions (action_size x action_samples x batch_size) and (1 x action_samples x batch_size) for `actions` and `logdpf` respectively.
"""
function (model::CovGaussianNetwork)(rng::AbstractRNG, state, action_samples::Int)
    batch_size = size(state, 3) #3
    x = model.pre(state)
    μ, cholesky_vec = model.μ(x), model.Σ(x)
    da = size(μ, 1)
    L = vec_to_tril(cholesky_vec, da)
    z = Zygote.ignore() do
        noise = randn(rng, eltype(μ), da, action_samples, batch_size)
        model.normalizer.(Flux.stack(map(.+, eachslice(μ, dims=3), eachslice(L, dims=3) .* eachslice(noise, dims=3)), 3))
    end
    logp_π = mvnormlogpdf(μ, L, z)
    return z, logp_π
end

function (model::CovGaussianNetwork)(state::AbstractArray, args...; kwargs...)
    model(Random.GLOBAL_RNG, state, args...; kwargs...)
end

"""
    (model::CovGaussianNetwork)(state, action)
    
Return the logpdf of the model sampling `action` when in `state`. 
State must be a 3D tensor with dimensions (state_size x 1 x batch_size).
Multiple actions may be taken per state, `action` must have dimensions (action_size x action_samples_per_state x batch_size)
Returns a 3D tensor with dimensions (1 x action_samples_per_state x batch_size)
"""
function (model::CovGaussianNetwork)(state::AbstractArray, action::AbstractArray)
    da = size(action, 1)
    x = model.pre(state)
    μ, cholesky_vec = model.μ(x), model.Σ(x)
    L = vec_to_tril(cholesky_vec, da)
    logp_π = mvnormlogpdf(μ, L, action)
    return logp_π
end

"""
If given 2D matrices as input, will return a 2D matrix of logpdf. States and actions are paired column-wise, one action per state.
"""
function (model::CovGaussianNetwork)(state::AbstractMatrix, action::AbstractMatrix)
    output = model(Flux.unsqueeze(state, 2), Flux.unsqueeze(action, 2))
    return dropdims(output, dims=2)
end

"""
Transform a vector containing the non-zero elements of a lower triangular da x da matrix into that matrix.
"""
function vec_to_tril(cholesky_vec, da)
    batch_size = size(cholesky_vec, 3)
    c2idx(i, j) = ((2da - j) * (j - 1)) ÷ 2 + i #return the position in cholesky_vec of the element of the triangular matrix at coordinates (i,j)
    function f(j) #return a slice (da x 1 x batchsize) containing the jth columns of the lower triangular cholesky decomposition of the covariance
        tc_diag = softplus.(cholesky_vec[c2idx(j, j):c2idx(j, j), :, :])
        tc_other = cholesky_vec[c2idx(j, j)+1:c2idx(j + 1, j + 1)-1, :, :]
        zs = Flux.Zygote.ignore() do
            zs = similar(cholesky_vec, da - size(tc_other, 1) - 1, 1, batch_size)
            zs .= zero(eltype(cholesky_vec))
            return zs
        end
        [zs; tc_diag; tc_other]
    end
    return mapreduce(f, hcat, 1:da)
end

#####
# DuelingNetwork
#####

"""
    DuelingNetwork(;base, val, adv)
    
Dueling network automatically produces separate estimates of the state value function network and advantage function network. The expected output size of val is 1, and adv is the size of the action space.
"""
Base.@kwdef struct DuelingNetwork{B,V,A}
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

#####
# PerturbationNetwork 
#####

"""
    PerturbationNetwork(;, ϕ)

Perturbation network outputs an adjustment to an action in the range [-ϕ, ϕ] to increase the diversity of seen actions.

# Keyword arguments
- `base`, a Flux based DNN model.
- `ϕ::Float32 = 0.05f0`
"""

Base.@kwdef struct PerturbationNetwork{N}
    base::N
    ϕ::Float32 = 0.05f0
end

Flux.@functor PerturbationNetwork

"""
This function accepts `state` and `action`, and then outputs actions after disturbance.
"""
function (model::PerturbationNetwork)(state, action)
    x = model.base(vcat(state, action))
    x = model.ϕ * tanh.(x)
    clamp.(x + action, -1.0f0, 1.0f0)
end

#####
# VAE (Variational Auto-Encoder)
#####

"""
    VAE(;encoder, decoder, latent_dims)
"""
Base.@kwdef struct VAE{E,D}
    encoder::E
    decoder::D
    latent_dims::Int
end

Flux.@functor VAE

function (model::VAE)(rng::AbstractRNG, state, action)
    μ, logσ = model.encoder(vcat(state, action))
    σ = exp.(logσ)
    z = μ .+ σ .* randn(rng, Float32, size(μ))
    u = decode(model, state, z)
    return u, μ, σ
end

function (model::VAE)(state, action)
    return model(Random.GLOBAL_RNG, state, action)
end

function decode(rng::AbstractRNG, model::VAE, state, z=nothing; is_normalize::Bool=true)
    if z === nothing
        z = clamp.(randn(rng, Float32, (model.latent_dims, size(state)[2:end]...)), -0.5f0, 0.5f0)
    end
    a = model.decoder(vcat(state, z))
    if is_normalize
        a = tanh.(a)
    end
    return a
end

function decode(model::VAE, state, z=nothing; is_normalize::Bool=true)
    decode(Random.GLOBAL_RNG, model, state, z; is_normalize)
end

function vae_loss(model::VAE, state, action)
    u, μ, σ = model(state, action)
    recon_loss = Flux.Losses.mse(u, action)
    kl_loss = -0.5f0 * mean(1.0f0 .+ log.(σ .^ 2) .- μ .^ 2 .- σ .^ 2)
    return recon_loss, kl_loss
end
