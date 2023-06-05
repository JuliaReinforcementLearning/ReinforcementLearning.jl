using Functors: @functor
import Flux
import Flux.onehotbatch
using ChainRulesCore: ignore_derivatives

#####
# ActorCritic
#####

export ActorCritic

"""
    ActorCritic(;actor, critic, optimizer=Adam())
The `actor` part must return logits (*Do not use softmax in the last layer!*), and the `critic` part must return a state value.
"""
Base.@kwdef struct ActorCritic{A,C,O}
    actor::A
    critic::C
end

@functor ActorCritic

#####
# GaussianNetwork
#####

export GaussianNetwork

"""
    GaussianNetwork(;pre=identity, μ, σ, min_σ=0f0, max_σ=Inf32)

Returns `μ` and `σ` when called.  Create a distribution to sample from using
`Normal.(μ, σ)`. `min_σ` and `max_σ` are used to clip the output from
`σ`. `pre` is a shared body before the two heads of the NN. σ should be > 0. 
You may enforce this using a `softplus` output activation. 
"""
Base.@kwdef struct GaussianNetwork{P,U,S}
    pre::P = identity
    μ::U
    σ::S
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
end

GaussianNetwork(pre, μ, σ) = GaussianNetwork(pre, μ, σ, 0.0f0, Inf32)

@functor GaussianNetwork

"""
This function is compatible with a multidimensional action space.

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal distribution. 
- `is_return_log_prob::Bool=false`, whether to calculate the conditional probability of getting actions in the given state.
"""
function (model::GaussianNetwork)(rng::AbstractRNG, s; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    x = model.pre(s)
    μ, raw_σ = model.μ(x), model.σ(x)
    σ = clamp.(raw_σ, model.min_σ, model.max_σ)
    if is_sampling
        z = ignore_derivatives() do
            noise = randn(rng, Float32, size(μ))
            μ .+ σ .* noise
        end
        if is_return_log_prob
            logp_π = sum(normlogpdf(μ, σ, z) .- (2.0f0 .* (log(2.0f0) .- z .- softplus.(-2.0f0 .* z))), dims=1)
            return z, logp_π
        else
            return z
        end
    else
        return μ, σ
    end
end

"""
    (model::GaussianNetwork)(rng::AbstractRNG, state::AbstractArray{<:Any, 3}, action_samples::Int)

Sample `action_samples` actions from each state. Returns a 3D tensor with dimensions `(action_size x action_samples x batch_size)`.
`state` must be 3D tensor with dimensions `(state_size x 1 x batch_size)`. Always returns the logpdf of each action along.
"""
function (model::GaussianNetwork)(rng::AbstractRNG, s::AbstractArray{<:Any, 3}, action_samples::Int)
    x = model.pre(s)
    μ, raw_σ = model.μ(x), model.σ(x)
    σ = clamp.(raw_σ, model.min_σ, model.max_σ)
    z = ignore_derivatives() do
        noise = randn(rng, Float32, (size(μ, 1), action_samples, size(μ, 3))...)
        μ .+ σ .* noise
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
    μ, raw_σ = model.μ(x), model.σ(x)
    σ = clamp.(raw_σ, model.min_σ, model.max_σ)
    logp_π = sum(normlogpdf(μ, σ, action) .- (2.0f0 .* (log(2.0f0) .- action .- softplus.(-2.0f0 .* action))), dims=1)
    return logp_π
end

#####
# CovGaussianNetwork
#####

export CovGaussianNetwork

"""
    CovGaussianNetwork(;pre=identity, μ, Σ)

Returns `μ` and `Σ` when called where μ is the mean and Σ is a covariance
matrix. Unlike GaussianNetwork, the output is 3-dimensional.  μ has dimensions
`(action_size x 1 x batch_size)` and Σ has dimensions `(action_size x action_size x
batch_size)`.  The Σ head of the `CovGaussianNetwork` should not directly return
a square matrix but a vector of length `action_size x (action_size + 1) ÷ 2`.
This vector will contain elements of the uppertriangular cholesky decomposition
of the covariance matrix, which is then reconstructed from it.  Sample from
`MvNormal.(μ, Σ)`.
"""
Base.@kwdef mutable struct CovGaussianNetwork{P,U,S}
    pre::P
    μ::U
    Σ::S
end

@functor CovGaussianNetwork

"""
    (model::CovGaussianNetwork)(rng::AbstractRNG, state::AbstractArray{<:Any, 3}; is_sampling::Bool=false, is_return_log_prob::Bool=false)

This function is compatible with a multidimensional action space. To work with covariance matrices, the outputs are 3D tensors.  If
sampling, return an actions tensor with dimensions `(action_size x action_samples
x batch_size)` and a `logp_π` tensor with dimensions `(1 x action_samples x batch_size)`. 
If not sampling, returns `μ`
with dimensions `(action_size x 1 x batch_size)` and `L`, the lower triangular of
the cholesky decomposition of the covariance matrix, with dimensions
`(action_size x action_size x batch_size)` The covariance matrices can be
retrieved with `Σ = stack(map(l -> l*l', eachslice(L, dims=3)); dims=3)`

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal
  distribution. 
- `is_return_log_prob::Bool=false`, whether to calculate the conditional
  probability of getting actions in the given state.
"""
function (model::CovGaussianNetwork)(rng::AbstractRNG, state::AbstractArray{<:Any, 3}; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    batch_size = size(state, 3)
    x = model.pre(state)
    μ, cholesky_vec = model.μ(x), model.Σ(x)
    da = size(μ, 1)
    L = vec_to_tril(cholesky_vec, da)

    if is_sampling
        z = ignore_derivatives() do
            noise = randn(rng, eltype(μ), da, 1, batch_size)
            z = copy(μ)
            for (m, l, n) in zip(eachslice(z, dims = 3), eachslice(L, dims=3), eachslice(noise, dims = 3))
                m .+= l * n
            end
            z
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
function (model::CovGaussianNetwork)(rng::AbstractRNG, state::AbstractVecOrMat; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    state_3d = reshape(state, size(state,1), 1, size(state, 2))
    output = model(rng, state_3d; is_sampling=is_sampling, is_return_log_prob=is_return_log_prob)
    if output isa Tuple && is_sampling
        dropdims(output[1], dims=2), dropdims(output[2], dims=2)
    elseif output isa Tuple
        dropdims(output[1], dims=2), output[2] #can't reduce the dims of the covariance tensor
    else
        dropdims(output, dims=2)
    end
end

"""
    (model::CovGaussianNetwork)(rng::AbstractRNG, state::AbstractArray{<:Any, 3}, action_samples::Int)

Sample `action_samples` actions per state in `state` and return the `actions,
logpdf(actions)`.  This function is compatible with a multidimensional action
space.  The outputs are 3D tensors with dimensions
`(action_size x action_samples x batch_size)` and `(1 x action_samples x
batch_size)` for `actions` and `logdpf` respectively.
"""
function (model::CovGaussianNetwork)(rng::AbstractRNG, state::AbstractArray{<:Any, 3}, action_samples::Int)
    batch_size = size(state, 3) #3
    x = model.pre(state)
    μ, cholesky_vec = model.μ(x), model.Σ(x)
    da = size(μ, 1)
    L = vec_to_tril(cholesky_vec, da)
    z = ignore_derivatives() do
        noise = randn(rng, eltype(μ), da, action_samples, batch_size)
        stack(map(.+, eachslice(μ, dims=3), eachslice(L, dims=3) .* eachslice(noise, dims=3)); dims=3)
    end
    logp_π = mvnormlogpdf(μ, L, z)
    return z, logp_π
end

function (model::CovGaussianNetwork)(state::AbstractArray, args...; kwargs...)
    model(Random.GLOBAL_RNG, state, args...; kwargs...)
end

"""
    (model::CovGaussianNetwork)(state::AbstractArray, action::AbstractArray)
    
Return the logpdf of the model sampling `action` when in `state`.  State must be
a 3D tensor with dimensions `(state_size x 1 x batch_size)`.  Multiple actions may
be taken per state, `action` must have dimensions `(action_size x
action_samples_per_state x batch_size)`. Returns a 3D tensor with dimensions `(1 x
action_samples_per_state x batch_size)`.
"""
function (model::CovGaussianNetwork)(state::AbstractArray{<:Any, 3}, action::AbstractArray{<:Any, 3})
    da = size(action, 1)
    x = model.pre(state)
    μ, cholesky_vec = model.μ(x), model.Σ(x)
    L = vec_to_tril(cholesky_vec, da)
    logp_π = mvnormlogpdf(μ, L, action)
    return logp_π
end

"""
If given 2D matrices as input, will return a 2D matrix of logpdf. States and
actions are paired column-wise, one action per state.
"""
function (model::CovGaussianNetwork)(state::AbstractMatrix, action::AbstractMatrix)
    output = model(Flux.unsqueeze(state, dims = 2), Flux.unsqueeze(action, dims=2))
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
        zs = ignore_derivatives() do
            zs = similar(cholesky_vec, da - size(tc_other, 1) - 1, 1, batch_size)
            zs .= zero(eltype(cholesky_vec))
            return zs
        end
        [zs; tc_diag; tc_other]
    end
    return mapreduce(f, hcat, 1:da)
end

#####
# DiscreteNetwork
#####

export CategoricalNetwork

"""
    CategoricalNetwork(model)([rng,] state::AbstractArray [, mask::AbstractArray{Bool}]; is_sampling::Bool=false, is_return_log_prob::Bool = false)

CategoricalNetwork wraps a model (typically a neural network) that takes a `state` input 
and outputs logits for a categorical distribution. The optional argument `mask` must be
an Array of `Bool` with the same size as `state` expect for the first dimension that must
have the length of the action vector. Actions mapped to `false` by mask have a logit equal to 
`-Inf` and/or a zero-probability of being sampled.

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal categorical distribution (returns a Flux.OneHotArray `z`). 
- `is_return_log_prob::Bool=false`, whether to return the *logits* (i.e. the unnormalized logprobabilities) of getting the sampled actions in the given state.
Only applies if `is_sampling` is true and will return `z, logits`.

If `is_sampling = false`, returns only the logits obtained by a simple forward pass into `model`.
"""
mutable struct CategoricalNetwork{P}
    model::P
end

@functor CategoricalNetwork

function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractArray; is_sampling::Bool=false, is_return_log_prob::Bool = false)
    logits = model.model(state) #may be 1-3 dimensional
    if is_sampling
        z = sample_categorical(rng,logits)
        if is_return_log_prob
            return z, logits
        else
            return z
        end
    else
        return logits
    end
end

function sample_categorical(rng, logits::AbstractArray)
    ignore_derivatives() do 
        log_probs = reshape(logsoftmax(logits, dims = 1), size(logits,1), :) # work in 2D
        gumbels = -log.(-log.(rand(rng, size(log_probs)...))) .+ log_probs # Gumbel-Max trick
        z = getindex.(argmax(gumbels, dims = 1), 1)
        reshape(onehotbatch(z, 1:size(logits,1)), size(logits)...) # reshape back to original shape
    end
end

function (model::CategoricalNetwork)(state::AbstractArray, args...; kwargs...)
    model(Random.GLOBAL_RNG, state, args...; kwargs...)
end

"""
    (model::CategoricalNetwork)([rng::AbstractRNG,] state::AbstractArray{<:Any, 3}, [mask::AbstractArray{Bool},] action_samples::Int)

Sample `action_samples` actions from each state. Returns a 3D tensor with dimensions `(action_size x action_samples x batch_size)`. 
Always returns the *logits* of each action along in a tensor with the same dimensions. The optional argument `mask` must be
an Array of `Bool` with the same size as `state` expect for the first dimension that must
have the length of the action vector. Actions mapped to `false` by mask have a logit equal to 
`-Inf` and/or a zero-probability of being sampled.
"""
function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractArray{<:Any, 3}, action_samples::Int)
    logits = model.model(state) #da x 1 x batch_size 
    z = ignore_derivatives() do 
        batch_size = size(state, 3) #3
        da = size(logits, 1)
        log_probs = logsoftmax(logits, dims = 1)
        gumbels = -log.(-log.(rand(rng, da, action_samples, batch_size))) .+ log_probs # Gumbel-Max trick
        z = getindex.(argmax(gumbels, dims = 1), 1)
        reshape(onehotbatch(z, 1:size(logits,1)), size(gumbels)...) # reshape to 3D due to onehotbatch behavior
    end   
    return z, reduce(hcat, Iterators.repeated(logits, action_samples))
end

function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractVecOrMat, action_samples::Int)
    model(rng, reshape(state, size(state, 1), 1, :), action_samples)
end

#Masked Methods

function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractArray, mask::AbstractArray{Bool}; is_sampling::Bool=false, is_return_log_prob::Bool = false)
    logits = model.model(state) #may be 1-3 dimensional
    logits .+= ifelse.(mask, 0f0, typemin(eltype(logits)))
    if is_sampling
        z = sample_categorical(rng,logits)
        if is_return_log_prob
            return z, logits
        else
            return z
        end
    else
        return logits
    end
end

function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractArray{<:Any, 3}, mask::AbstractArray{Bool, 3}, action_samples::Int)
    logits = model.model(state) #da x 1 x batch_size 
    logits .+= ifelse.(mask, 0f0, typemin(eltype(logits)))
    z = ignore_derivatives() do 
        batch_size = size(state, 3) #3
        da = size(logits, 1)
        log_probs = logsoftmax(logits, dims = 1)
        gumbels = -log.(-log.(rand(rng, da, action_samples, batch_size))) .+ log_probs # Gumbel-Max trick
        z = getindex.(argmax(gumbels, dims = 1), 1)
        reshape(onehotbatch(z, 1:size(logits,1)), size(gumbels)...) # reshape to 3D due to onehotbatch behavior
    end   
    return z, reduce(hcat, Iterators.repeated(logits, action_samples))
end

function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractArray, mask::AbstractArray, action_samples::Int)
    model(rng, reshape(state, size(state, 1), 1, :), reshape(mask, size(mask, 1), 1, :), action_samples)
end

#####
# DuelingNetwork
#####

export DuelingNetwork

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

export PerturbationNetwork

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

export VAE

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
    μ, σ = model.encoder(vcat(state, action))
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

#####
# TwinNetwork
#####

export TwinNetwork

Base.@kwdef mutable struct TwinNetwork{S,T}
    source::S
    target::T
    sync_freq::Int = 1
    ρ::Float32 = 0.0f0
    n_optimise::Int = 0
end

TwinNetwork(x; kw...) = TwinNetwork(; source=x, target=deepcopy(x), kw...)

@functor TwinNetwork (source, target)

Flux.trainable(model::TwinNetwork) = (model.source,)

(model::TwinNetwork)(args...) = model.source(args...)

function RLBase.optimise!(A::Approximator{<:TwinNetwork}, ::PostActStage, gs)
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)
    M = A.model
    M.n_optimise += 1

    if M.n_optimise % M.sync_freq == 0
        # polyak averaging
        for (dest, src) in zip(Flux.params(M.target), Flux.params(M.source))
            dest .= M.ρ .* dest .+ (1 - M.ρ) .* src
        end
    end
end
