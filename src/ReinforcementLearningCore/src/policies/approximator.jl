export Approximator, TargetNetwork

using Flux

Base.@kwdef mutable struct Approximator{M,O}
    model::M
    optimiser::O
end

Base.show(io::IO, m::MIME"text/plain", A::Approximator) = show(io, m, convert(AnnotatedStructTree, A))

@functor Approximator (model,)

forward(A::Approximator, args...; kwargs...) = A.model(args...; kwargs...)

RLBase.optimise!(A::Approximator, gs) = Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)

target(ap::Approximator) = ap.model #see TargetNetwork

"""
    TargetNetwork(network::Approximator; sync_freq::Int = 1, ρ::Float32 = 0f0)

An extended Approximator that holds a target for the network to be updated towards. 
This is typically to stabilize Approximators that learn with a temporal difference
loss, such as state or action-value networks. `sync_freq` is the number of updates of
`network` between each update of the `target`. ρ (\rho) is "how much of the target is kept
when updating it". The two common usages of TargetNetwork are 
- use ρ = 0 to totally replace `target` with `network` every sync_freq updates.
- use ρ < 1 (but close to one) and sync_freq = 1 to let the target follow `network` with polyak averaging.

Note to developpers: `target(::TargetNetwork)` will return the target model and 
`target(::Approximator)` returns the model. You can therefore use this interface 
to create agents agnostic to whether the model has a target or not.
"""
Base.@kwdef mutable struct TargetNetwork{M}
    source::Approximator{M}
    target::M
    sync_freq::Int = 1
    ρ::Float32 = 0.0f0
    n_optimise::Int = 0
end

function TargetNetwork(x; kw...) 
    if haskey(kw, :ρ)
        @assert 0 <= kw[:ρ] <= 1 "ρ must in [0,1]"
    end
    TargetNetwork(; source=x, target=deepcopy(x), kw...)
end

@functor TargetNetwork (source, target)

Flux.trainable(model::TargetNetwork) = (model.source,)

forward(tn::TargetNetwork, args...) = forward(tn.model, args...)

target(tn::TargetNetwork) = tn.target

function RLBase.optimise!(tn::TargetNetwork, gs)
    A = tn.source
    Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)
    tn.n_optimise += 1

    if M.n_optimise % M.sync_freq == 0
        # polyak averaging
        for (dest, src) in zip(Flux.params(tn.target.model), Flux.params(tn.model))
            dest .= tn.ρ .* dest .+ (1 - tn.ρ) .* src
        end
        tn.n_optimise = 0
    end
end
