export VPG

using Random: GLOBAL_RNG, shuffle
using Distributions: ContinuousDistribution, DiscreteDistribution, logpdf
using Functors: @functor
using Flux: params, softmax, gradient, logsoftmax
using StatsBase: mean
using ChainRulesCore: ignore_derivatives

"""
Vanilla Policy Gradient
"""
Base.@kwdef struct VPG{A,B,D} <: AbstractPolicy
    "For discrete actions, logits before softmax is expected. For continuous actions, a `Tuple` of arguments are expected to initialize `dist`"
    approximator::A
    baseline::B = nothing
    "`ContinuousDistribution` or `DiscreteDistribution`"
    dist::D
    "discount ratio"
    γ::Float32 = 0.99f0
    batch_size::Int = 1024
    rng::AbstractRNG = GLOBAL_RNG
end

@functor VPG (approximator, baseline)

function (π::VPG)(env::AbstractEnv)
    res = env |> state |> send_to_device(π) |> π.approximator |> send_to_host
    rand(π.rng, action_distribution(π.dist, res), 1)
end

function (p::Agent{<:VPG})(::PostEpisodeStage, env::AbstractEnv)
    p.trajectory.container[] = true
    optimise!(p.policy, p.trajectory.container)
    empty!(p.trajectory.container)
end

RLBase.optimise!(::Agent{<:VPG}) = nothing

function RLBase.optimise!(π::VPG, episode::Episode)
    gain = discount_rewards(episode[:reward][:], π.γ)
    for inds in Iterators.partition(shuffle(π.rng, 1:length(episode)), π.batch_size)
        optimise!(π, (state=episode[:state][inds], action=episode[:action][inds], gain=gain[inds]))
    end
end

function RLBase.optimise!(p::VPG, batch::NamedTuple{(:state, :action, :gain)})
    A = p.approximator
    B = p.baseline
    s, a, g = map(Array, batch) # !!! FIXME

    if isnothing(B)
        δ = normalise(g)
    else
        gs = gradient(params(B)) do
            δ = g - vec(B(s))
            loss = mean(δ .^ 2)
            ignore_derivatives() do
                # @info "VPG/baseline" loss = loss δ
            end
            loss
        end
        optimise!(B, gs)
    end

    gs = gradient(params(A)) do
        log_probₐ = logpdf.(action_distribution(p.dist, A(s)), a)
        loss = -mean(log_probₐ .* δ)
        ignore_derivatives() do
            # @info "VPG" loss = loss
        end
        loss
    end
    optimise!(A, gs)
end
