export Approximator, TargetNetwork, target, model

using Flux


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
mutable struct TargetNetwork{M}
    network::Approximator{M}
    target::M
    sync_freq::Int
    ρ::Float32
    n_optimise::Int
end

function TargetNetwork(network; sync_freq = 1, ρ = 0f0)
    @assert 0 <= ρ <= 1 "ρ must in [0,1]"
    # NOTE: model is pushed to gpu in Approximator, need to transfer to cpu before deepcopy, then push target model to gpu
    target = gpu(deepcopy(cpu(network.model)))
    TargetNetwork(network, target, sync_freq, ρ, 0)
end

@functor TargetNetwork (network, target)

Flux.trainable(model::TargetNetwork) = (model.network,)

forward(tn::TargetNetwork, args...) = forward(tn.network, args...)

model(tn::TargetNetwork) = model(tn.network)
target(tn::TargetNetwork) = tn.target

function RLBase.optimise!(tn::TargetNetwork, grad)
    A = tn.network
    optimise!(A, grad)

    tn.n_optimise += 1

    if tn.n_optimise % tn.sync_freq == 0
        # polyak averaging
        for (dest, src) in zip(Flux.params(target(tn)), Flux.params(tn.network))
            dest .= tn.ρ .* dest .+ (1 - tn.ρ) .* src
        end
        tn.n_optimise = 0
    end
end
