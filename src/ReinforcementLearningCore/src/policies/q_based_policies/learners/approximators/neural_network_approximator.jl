export NeuralNetworkApproximator, ActorCritic, GaussianNetwork, DuelingNetwork

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
    GaussianNetwork(;pre=identity, μ, logσ)

Returns `μ` and `logσ` when called. 
Create a distribution to sample from 
using `Normal.(μ, exp.(logσ))`.
"""
Base.@kwdef struct GaussianNetwork{P,U,S}
    pre::P = identity
    μ::U
    logσ::S
end

Flux.@functor GaussianNetwork

"""
This function is compatible with a multidimensional action space. When outputting an action, it uses `tanh` to normalize it.

- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `is_sampling::Bool=false`, whether to sample from the obtained normal distribution. 
- `is_return_log_prob::Bool=false`, whether to calculate the conditional probability of getting actions in the given state.
"""
function (model::GaussianNetwork)(rng::AbstractRNG, state; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    x = model.pre(state)
    μ, logσ = model.μ(x), model.logσ(x) 
    if is_sampling
        π_dist = Normal.(μ, exp.(logσ))
        z = rand.(rng, π_dist)
        if is_return_log_prob
            logp_π = sum(logpdf.(π_dist, z) .- (2.0f0 .* (log(2.0f0) .- z .- softplus.(-2.0f0 .* z))), dims = 1)
            return tanh.(z), logp_π
        else
            return tanh.(z)
        end
    else
        return μ, logσ
    end
end

function (model::GaussianNetwork)(state; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    model(Random.GLOBAL_RNG, state; is_sampling=is_sampling, is_return_log_prob=is_return_log_prob)
end

#####
# DuelingNetwork
#####

"""
    DuelingNetwork(;base, val, adv)
    
Dueling network automatically produces separate estimates of the state value function network and advantage function network. The expected output size of val is 1, and adv is the size of the action space.
"""
struct DuelingNetwork{B,V,A}
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