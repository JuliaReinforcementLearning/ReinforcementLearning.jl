export Approximator, TargetNetwork, target, model

using Flux


"""
    Approximator(model, optimiser)

Wraps a Flux trainable model and implements the `RLBase.optimise!(::Approximator, ::Gradient)` 
interface. See the RLCore documentation for more information on proper usage.
"""
Base.@kwdef mutable struct Approximator{M,O}
    model::M
    optimiser::O
end

Base.show(io::IO, m::MIME"text/plain", A::Approximator) = show(io, m, convert(AnnotatedStructTree, A))

@functor Approximator (model,)

forward(A::Approximator, args...; kwargs...) = A.model(args...; kwargs...)

RLBase.optimise!(A::Approximator, gs) = Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)

target(ap::Approximator) = ap.model #see TargetNetwork
model(ap::Approximator) = ap.model #see TargetNetwork

"""
    TargetNetwork(network::Approximator; sync_freq::Int = 1, ρ::Float32 = 0f0)

Wraps an Approximator to hold a target network that is updated towards the model of the 
approximator. 
- `sync_freq` is the number of updates of `network` between each update of the `target`. 
- ρ (\rho) is "how much of the target is kept when updating it". 

The two common usages of TargetNetwork are 
- use ρ = 0 to totally replace `target` with `network` every sync_freq updates.
- use ρ < 1 (but close to one) and sync_freq = 1 to let the target follow `network` with polyak averaging.

Implements the `RLBase.optimise!(::TargetNetwork, ::Gradient)` interface to update the model with the gradient
and the target with weights replacement or Polyak averaging.

Note to developpers: `model(::TargetNetwork)` will return the trainable Flux model 
and `target(::TargetNetwork)` returns the target model and `target(::Approximator)`
returns the non-trainable Flux model. See the RLCore documentation.
"""
Base.@kwdef mutable struct TargetNetwork{M}
    network::Approximator{M}
    target::M
    sync_freq::Int = 1
    ρ::Float32 = 0.0f0
    n_optimise::Int = 0
end

function TargetNetwork(x; kw...) 
    if haskey(kw, :ρ)
        @assert 0 <= kw[:ρ] <= 1 "ρ must in [0,1]"
    end
    TargetNetwork(; network=x, target=deepcopy(x.model), kw...)
end

@functor TargetNetwork (network, target)

Flux.trainable(model::TargetNetwork) = (model.network,)

forward(tn::TargetNetwork, args...) = forward(tn.network, args...)

model(tn::TargetNetwork) = model(tn.network)
target(tn::TargetNetwork) = tn.target

function RLBase.optimise!(tn::TargetNetwork, gs)
    A = tn.network
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)
    tn.n_optimise += 1

    if tn.n_optimise % tn.sync_freq == 0
        # polyak averaging
        for (dest, src) in zip(Flux.params(target(tn)), Flux.params(tn.network))
            dest .= tn.ρ .* dest .+ (1 - tn.ρ) .* src
        end
        tn.n_optimise = 0
    end
end
