export TRPO

using Random: GLOBAL_RNG, shuffle, AbstractRNG
using Functors: @functor
using Zygote: forward_jacobian
using Flux: Flux, destructure
using StatsBase: mean
using Distributions: kldivergence, pdf
using ChainRulesCore: ignore_derivatives
using ForwardDiff


"""
Trust Region Policy Optimization
"""
Base.@kwdef struct TRPO{A,B,D} <: AbstractPolicy
    approximator::A
    baseline::B
    "only a discrete action space is supported for now"
    dist::D = Distributions.Categorical
    γ::Float32 = 0.99f0
    batch_size::Int = 1024
    rng::AbstractRNG = GLOBAL_RNG
    max_backtrack_step::Int = 10
    kldivergence_limit::Float32 = 1f-2
    backtrack_coeff::Float32 = 0.8f0
end

IsPolicyGradient(::Type{<:TRPO}) = IsPolicyGradient()
@functor TRPO (approximator, baseline)

function (π::TRPO)(env::AbstractEnv)
    res = env |> state |> send_to_device(π) |> π.approximator |> send_to_host
    rand(π.rng, action_distribution(π.dist, res)[1])
end

function (p::Agent{<:TRPO})(::PostEpisodeStage, env::AbstractEnv)
    p.trajectory.container[] = true
    optimise!(p.policy, p.trajectory.container)
    empty!(p.trajectory.container)
end

RLBase.optimise!(::Agent{<:TRPO}) = nothing

function RLBase.optimise!(π::TRPO, episode::Episode)
    gain = discount_rewards(episode[:reward][:], π.γ)
    println("episode reward is $(gain[1])")
    for inds in Iterators.partition(shuffle(π.rng, 1:length(episode)), π.batch_size)
        optimise!(π, (state=episode[:state][inds], action=episode[:action][inds], gain=gain[inds]))
    end
end

function RLBase.optimise!(p::TRPO, batch::NamedTuple{(:state, :action, :gain)})
    A = p.approximator
    B = p.baseline
    s, a, g = map(Array, batch) # !!! FIXME

    # fit value network on mean-square
    # store δ for advantage estimate
    if isnothing(B)
        δ = normalise(g)
    else
        gs = gradient(params(B)) do
            δ = g - vec(B(s))
            loss = mean(δ .^ 2)
            ignore_derivatives() do
                # @info "TRPO/baseline" loss = loss δ
            end
            loss
        end
        optimise!(B, gs)
    end

    old_logits = Ref{Matrix{Float32}}()

    gps = gradient(params(A.model)) do
        old_logits[] = A.model(s)
        total_loss = map(eachcol(softmax(old_logits[])), a) do x, y
            x[y]
        end .* δ
        loss = mean(total_loss)
        loss
    end

    ĝₖ = mapreduce(vec, vcat, gps)

    #println("action parameters are $(old_logits[]), ĝₖ = $ĝₖ")

    θₖ, re = Flux.destructure(A.model)

    hes1(x) = hvp_direct(A.model, s, old_logits[], x)

    x̂ₖ = conjugate_gradient(hes1, ĝₖ)

    # backtracking line search
    search_length = sqrt(2*p.kldivergence_limit / (x̂ₖ' * ĝₖ)) .* x̂ₖ
    println("search_length is $search_length")
    any(isnan, search_length) && error("search length is nan, x̂ₖ = $x̂ₖ, ĝₖ = $ĝₖ")
    search_condition(θ) = begin
        model_θ = re(θ)
        sur_adv = surrogate_advantage(model_θ, s, a, δ, old_logits[]) - mean(δ)
        println("surrogate advantage is $sur_adv")
        kld_excess = kld_direct(model_θ, s, old_logits[]) - p.kldivergence_limit
        println("kldivergence excess is $kld_excess")
        sur_adv > 0 && kld_excess <= 0
    end

    θₖ₊₁ = backtrack_line_search(θₖ, p.backtrack_coeff, search_length, p.max_backtrack_step, search_condition)

    # update policy approximator
    ps = params(A.model)
    copy!(ps, θₖ₊₁)
end
