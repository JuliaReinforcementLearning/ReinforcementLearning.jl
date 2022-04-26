import Distributions.Discrete
import Flux.softmax, Flux.onehotbatch, Flux.logsoftmaxd
using Flux, Distributions, Random

mutable struct CategoricalNetwork{P}
    model::P
end

Flux.@functor CategoricalNetwork

function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractArray; is_sampling::Bool=false)
    logits = model.model(state)
    if is_sampling
        z = Flux.Zygote.ignore() do 
            probs = reshape(softmax(logits, dims = 1), size(logits,1), :)
            dists = [Categorical(x; check_args = false) for x in eachcol(probs)]
            z = rand.(rng, dists)
            reshape(onehotbatch(z, 1:size(logits,1)), size(logits)...)
        end
        return z, logits
    else
        return logits
    end
end

function (model::CategoricalNetwork)(state::AbstractArray, args...; kwargs...)
    model(Random.GLOBAL_RNG, state, args...; kwargs...)
end

function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractArray, action_samples::Int)
    batch_size = size(state, 3) #3
    logits = model.model(state)
    da = size(logits, 1)
    z = Flux.Zygote.ignore() do 
        probs = reshape(softmax(logits, dims = 1), size(logits,1), :)
        dists = [Categorical(x; check_args = false) for x in eachcol(probs)]
        z = reshape(reduce(hcat, rand.(rng, dists, action_samples)), 1, action_samples, batch_size)
        reshape(onehotbatch(z, 1:size(logits,1)), da, action_samples, batch_size)
    end
    return z, reduce(hcat, repeat(logits, outer = (1, action_samples)))
end

function (model::CategoricalNetwork)(rng::AbstractRNG, state::AbstractMatrix, action_samples::Int)
    model(rng, reshape(state, size(state, 1), 1, :), action_samples)
end

#multi action sampling