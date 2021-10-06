export NeuralNetworkApproximator,
    ActorCritic, GaussianNetwork, DuelingNetwork, PerturbationNetwork
export VAE, decode, vae_loss

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
    (model = x.model,), y -> NeuralNetworkApproximator(y.model, x.optimizer)

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
    (actor = x.actor, critic = x.critic), y -> ActorCritic(y.actor, y.critic, x.optimizer)

RLBase.update!(app::ActorCritic, gs) = Flux.Optimise.update!(app.optimizer, params(app), gs)

function Base.copyto!(dest::ActorCritic, src::ActorCritic)
    Flux.loadparams!(dest.actor, params(src.actor))
    Flux.loadparams!(dest.critic, params(src.critic))
end

#####
# GaussianNetwork
#####

"""
    GaussianNetwork(;pre=identity, μ, logσ, min_σ=0f0, max_σ=Inf32)

Returns `μ` and `logσ` when called.  Create a distribution to sample from using
`Normal.(μ, exp.(logσ))`. `min_σ` and `max_σ` are used to clip the output from
`logσ`.
"""
Base.@kwdef struct GaussianNetwork{P,U,S}
    pre::P = identity
    μ::U
    logσ::S
    min_σ::Float32 = 0.0f0
    max_σ::Float32 = Inf32
end

Flux.@functor GaussianNetwork

"""
This function is compatible with a multidimensional action space. When outputting an action, it uses `tanh` to normalize it.

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal distribution. 
- `is_return_log_prob::Bool=false`, whether to calculate the conditional probability of getting actions in the given state.
"""
function (model::GaussianNetwork)(
    rng::AbstractRNG,
    state;
    is_sampling::Bool = false,
    is_return_log_prob::Bool = false,
)
    x = model.pre(state)
    μ, raw_logσ = model.μ(x), model.logσ(x)
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))
    if is_sampling
        σ = exp.(logσ)
        z = μ .+ σ .* send_to_device(device(model), randn(rng, Float32, size(μ)))
        if is_return_log_prob
            logp_π = sum(
                normlogpdf(μ, σ, z) .-
                (2.0f0 .* (log(2.0f0) .- z .- softplus.(-2.0f0 .* z))),
                dims = 1,
            )
            return tanh.(z), logp_π
        else
            return tanh.(z)
        end
    else
        return μ, logσ
    end
end

function (model::GaussianNetwork)(
    state;
    is_sampling::Bool = false,
    is_return_log_prob::Bool = false,
)
    model(
        Random.GLOBAL_RNG,
        state;
        is_sampling = is_sampling,
        is_return_log_prob = is_return_log_prob,
    )
end

function (model::GaussianNetwork)(state, action)
    x = model.pre(state)
    μ, raw_logσ = model.μ(x), model.logσ(x)
    logσ = clamp.(raw_logσ, log(model.min_σ), log(model.max_σ))
    σ = exp.(logσ)
    logp_π = sum(
        normlogpdf(μ, σ, action) .-
        (2.0f0 .* (log(2.0f0) .- action .- softplus.(-2.0f0 .* action))),
        dims = 1,
    )
    return logp_π
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
    return val .+ m.adv(x) .- mean(m.adv(x), dims = 1)
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
    z = μ .+ σ .* send_to_device(device(model), randn(rng, Float32, size(μ)))
    u = decode(model, state, z)
    return u, μ, σ
end

function (model::VAE)(state, action)
    return model(Random.GLOBAL_RNG, state, action)
end

function decode(rng::AbstractRNG, model::VAE, state, z = nothing; is_normalize::Bool = true)
    if z === nothing
        z =
            clamp.(
                randn(rng, Float32, (model.latent_dims, size(state)[2:end]...)),
                -0.5f0,
                0.5f0,
            )
        z = send_to_device(device(model), z)
    end
    a = model.decoder(vcat(state, z))
    if is_normalize
        a = tanh.(a)
    end
    return a
end

function decode(model::VAE, state, z = nothing; is_normalize::Bool = true)
    decode(Random.GLOBAL_RNG, model, state, z; is_normalize)
end

function vae_loss(model::VAE, state, action)
    u, μ, σ = model(state, action)
    recon_loss = Flux.Losses.mse(u, action)
    kl_loss = -0.5f0 * mean(1.0f0 .+ log.(σ .^ 2) .- μ .^ 2 .- σ .^ 2)
    return recon_loss, kl_loss
end
