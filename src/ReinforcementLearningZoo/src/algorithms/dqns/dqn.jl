export DQNLearner

mutable struct DQNLearner{A<:Approximator{<:TwinNetwork}} <: Any
    approximator::A
    loss_func::Any
    γ::Float32 = 0.99f0
    rng::AbstractRNG = Random.GLOBAL_RNG
    # for logging
    loss::Float32 = 0.0f0
end

Functors.functor(x::DQNLearner) = (; approximator=x.approximator), y -> @set x.approximator = y.approximator

function RLBase.optimise!(learner::DQNLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
    γ = learner.sampler.γ
    loss_func = learner.loss_func
    n = learner.sampler.n
    batch_size = learner.sampler.batch_size
    is_enable_double_DQN = learner.is_enable_double_DQN
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    q_values = Q(s′)

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    selected_actions = dropdims(argmax(q_values, dims=1), dims=1)
    q′ = Qₜ(s′)[selected_actions]

    G = r .+ γ^n .* (1 .- t) .* q′

    gs = gradient(params(Q)) do
        q = Q(s)[a]
        loss = loss_func(G, q)
        ignore() do
            learner.loss = loss
        end
        loss
    end

    update!(Q, gs)
end
