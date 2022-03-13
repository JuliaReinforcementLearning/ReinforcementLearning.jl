export VPGPolicy

"""
Vanilla Policy Gradient

VPGPolicy(;kwargs)

# Keyword arguments
- `approximator`,
- `baseline`,
- `dist`, distribution function of the action
- `γ`, discount factor
- `α_θ`, step size of policy parameter
- `α_w`, step size of baseline parameter
- `batch_size`,
- `rng`,
- `loss`,
- `baseline_loss`,


if the action space is continuous,
then the env should transform the action value, (such as using tanh),
in order to make sure low ≤ value ≤ high
"""
Base.@kwdef mutable struct VPGPolicy{
    A<:NeuralNetworkApproximator,
    B<:Union{NeuralNetworkApproximator,Nothing},
    S,
    R<:AbstractRNG,
} <: AbstractPolicy
    approximator::A
    baseline::B = nothing
    action_space::S
    dist::Any
    γ::Float32 = 0.99f0 # discount factor
    α_θ = 1.0f0 # step size of policy
    α_w = 1.0f0 # step size of baseline
    batch_size::Int = 1024
    rng::R = Random.GLOBAL_RNG
    loss::Float32 = 0.0f0
    baseline_loss::Float32 = 0.0f0
end

"""
About continuous action space, see
* [Diagonal Gaussian Policies](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
* [Clipped Action Policy Gradient](https://arxiv.org/pdf/1802.07564.pdf)
"""

function (π::VPGPolicy)(env::AbstractEnv)
    to_dev(x) = send_to_device(device(π.approximator), x)

    logits = env |> state |> to_dev |> π.approximator |> send_to_host

    if π.action_space isa AbstractVector
        dist = logits |> softmax |> π.dist
        action = π.action_space[rand(π.rng, dist)]
    elseif π.action_space isa Interval
        dist = π.dist.(logits...)
        action = rand.(π.rng, dist)[1]
    else
        error("not implemented")
    end
    action
end

function (π::VPGPolicy)(env::MultiThreadEnv)
    error("not implemented")
    # TODO: can PG support multi env? PG only get updated at the end of an episode.
end

function RLBase.update!(
    trajectory::ElasticSARTTrajectory,
    policy::VPGPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(trajectory[:state], state(env))
    push!(trajectory[:action], action)
end

function RLBase.update!(
    t::ElasticSARTTrajectory,
    ::VPGPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end

RLBase.update!(::VPGPolicy, ::ElasticSARTTrajectory, ::AbstractEnv, ::PreActStage) = nothing

function RLBase.update!(
    π::VPGPolicy,
    traj::ElasticSARTTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage,
)
    model = π.approximator
    to_dev(x) = send_to_device(device(model), x)

    states = traj[:state]
    actions = traj[:action] |> Array # need to convert ElasticArray to Array, or code will fail on gpu. `log_prob[CartesianIndex.(A, 1:length(A))`
    gains = traj[:reward] |> x -> discount_rewards(x, π.γ)

    for idx in Iterators.partition(shuffle(1:length(traj[:terminal])), π.batch_size)
        S = select_last_dim(states, idx) |> Array |> to_dev
        A = actions[idx]
        G = gains[idx] |> x -> Flux.unsqueeze(x, 1) |> to_dev
        # gains is a 1 column array, but the output of flux model is 1 row, n_batch columns array. so unsqueeze it.

        if π.baseline isa NeuralNetworkApproximator
            gs = gradient(Flux.params(π.baseline)) do
                δ = G - π.baseline(S)
                loss = mean(δ .^ 2) * π.α_w # mse
                ignore() do
                    π.baseline_loss = loss
                end
                loss
            end
            update!(π.baseline, gs)
        elseif π.baseline isa Nothing
            # Normalization. See
            # (http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/hw2_final.pdf)
            # (https://web.stanford.edu/class/cs234/assignment3/solution.pdf)
            # normalise should not be used with baseline. or the loss of the policy will be too small.
            δ = G |> x -> normalise(x; dims = 2)
        end

        gs = gradient(Flux.params(model)) do
            if π.action_space isa AbstractVector
                log_prob = S |> model |> logsoftmax
                log_probₐ = log_prob[CartesianIndex.(A, 1:length(A))]
            elseif π.action_space isa Interval
                dist = π.dist.(model(S)...) # TODO: this part does not work on GPU. See: https://github.com/JuliaStats/Distributions.jl/issues/1183 .
                log_probₐ = logpdf.(dist, A)
            end
            loss = -mean(log_probₐ .* δ) * π.α_θ
            ignore() do
                π.loss = loss
            end
            loss
        end
        update!(model, gs)
    end
end
