export QRDQNLearner, quantile_huber_loss

function quantile_huber_loss(ŷ, y; κ = 1.0f0)
    N, B = size(y)
    Δ = reshape(y, N, 1, B) .- reshape(ŷ, 1, N, B)
    abs_error = abs.(Δ)
    quadratic = min.(abs_error, κ)
    linear = abs_error .- quadratic
    huber_loss = 0.5f0 .* quadratic .* quadratic .+ κ .* linear

    cum_prob = send_to_device(device(y), range(0.5f0 / N; length = N, step = 1.0f0 / N))
    loss = Zygote.dropgrad(abs.(cum_prob .- (Δ .< 0))) .* huber_loss
    mean(sum(loss; dims = 1))
end

mutable struct QRDQNLearner{Tq<:AbstractApproximator,Tt<:AbstractApproximator,Tf,R} <:
               AbstractLearner
    approximator::Tq
    target_approximator::Tt
    min_replay_history::Int
    update_freq::Int
    update_step::Int
    target_update_freq::Int
    sampler::NStepBatchSampler
    n_quantile::Int
    loss_func::Tf
    rng::R
    # for recording
    loss::Float32
end

"""
    QRDQNLearner(;kwargs...)

See paper: [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/pdf/1710.10044.pdf)

# Keywords

- `approximator`::[`AbstractApproximator`](@ref): used to get quantile-values of a batch of states. The output should be of size `(n_quantile, n_action)`.
- `target_approximator`::[`AbstractApproximator`](@ref): similar to `approximator`, but used to estimate the quantile values of the next state batch.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=1`: the frequency of updating the `approximator`.
- `n_quantile::Int=1`: the number of quantiles.
- `target_update_freq::Int=100`: the frequency of syncing `target_approximator`.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.
- `traces = SARTS`, set to `SLARTSL` if you are to apply to an environment of `FULL_ACTION_SET`.
- `loss_func`=[`quantile_huber_loss`](@ref).
"""
function QRDQNLearner(;
    approximator,
    target_approximator,
    stack_size::Union{Int,Nothing} = nothing,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    update_freq::Int = 1,
    n_quantile::Int = 1,
    target_update_freq::Int = 100,
    traces = SARTS,
    update_step = 0,
    loss_func = quantile_huber_loss,
    rng = Random.GLOBAL_RNG,
)
    copyto!(approximator, target_approximator)
    sampler = NStepBatchSampler{traces}(;
        γ = γ,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )

    N = n_quantile

    QRDQNLearner(
        approximator,
        target_approximator,
        min_replay_history,
        update_freq,
        update_step,
        target_update_freq,
        sampler,
        N,
        loss_func,
        rng,
        0.0f0,
    )
end

Flux.functor(x::QRDQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

function (learner::QRDQNLearner)(env)
    s = send_to_device(device(learner.approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    q = reshape(learner.approximator(s), learner.n_quantile, :)
    vec(mean(q, dims = 1)) |> send_to_host
end

function RLBase.update!(learner::QRDQNLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
    γ = learner.sampler.γ
    n = learner.sampler.n
    batch_size = learner.sampler.batch_size
    N = learner.n_quantile
    D = device(Q)
    loss_func = learner.loss_func

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    target_quantiles = reshape(Qₜ(s′), N, :, batch_size)
    qₜ = dropdims(mean(target_quantiles; dims = 1); dims = 1)
    aₜ = dropdims(argmax(qₜ, dims = 1); dims = 1)
    @views target_quantile_aₜ = target_quantiles[:, aₜ]
    y =
        reshape(r, 1, batch_size) .+
        γ .* reshape(1 .- t, 1, batch_size) .* target_quantile_aₜ

    gs = gradient(params(Q)) do
        q = reshape(Q(s), N, :, batch_size)
        @views ŷ = q[:, a]

        loss = loss_func(ŷ, y)

        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end
