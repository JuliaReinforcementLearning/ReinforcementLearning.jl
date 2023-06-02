"""
    NFQ{A<:AbstractApproximator, F, R} <: AbstractLearner
    NFQ(approximator::A, num_iterations::Integer epochs::Integer, loss_function::F, batch_size::Integer, rng::R, γ::Float32) where {A<:AbstractApproximator, F, R}
Neural Fitted Q-iteration as implemented in [1]

# Keyword arguments
- `approximator::AbstractApproximator` neural network
- `num_iterations::Integer` number of value iteration iterations in FQI loop (i.e. the outer loop)
- `epochs` number of epochs to train neural network per iteration
- `loss_function::F` loss function of the NN
- `sampler::BatchSampler{SARTS}` data sampler
- `rng::R` random number generator
- `γ::Float32` discount rate

# References
[1] Riedmiller, M. (2005). Neural Fitted Q Iteration – First Experiences with a Data Efficient Neural Reinforcement Learning Method. In: Gama, J., Camacho, R., Brazdil, P.B., Jorge, A.M., Torgo, L. (eds) Machine Learning: ECML 2005. ECML 2005. Lecture Notes in Computer Science(), vol 3720. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11564096_32
"""
Base.@kwdef struct NFQ{A<:AbstractApproximator, F, R} <: AbstractLearner
    approximator::A
    num_iterations::Integer = 20
    epochs::Integer = 100
    loss_function::F = mse
    sampler::BatchSampler{SARTS} = BatchSampler(32)
    rng::R = Random.GLOBAL_RNG
    γ::Float32 = 0.9f0
end

function NFQ(;
    approximator::A,
    num_iterations::Integer = 20,
    epochs::Integer = 1000,
    loss_function::F = mse,
    batch_size::Integer=32,
    rng=Random.GLOBAL_RNG,
    γ::Float32 = 0.9f0,
    ) where {A<:AbstractApproximator, F}
    NFQ(approximator, num_iterations, epochs, loss_function, BatchSampler{SARTS}(batch_size), rng, γ)
end

# Copied from BasicDQN but sure whether it's appropriate
Flux.functor(x::NFQ) = (Q = x.approximator,), y -> begin
    x = @set x.approximator = y.Q
    x
end

function (learner::NFQ)(env)
    as = action_space(env)
    return vcat(repeat(state(env), inner=(1, length(as))), transpose(as)) |> x -> send_to_device(device(learner.approximator), x) |> learner.approximator |> send_to_host |> vec
end

function RLBase.update!(learner::NFQ, traj::AbstractTrajectory, ::AbstractEnv, ::PreExperimentStage) end
function RLBase.update!(learner::NFQ, traj::AbstractTrajectory, ::AbstractEnv, ::PreActStage) end
function RLBase.update!(learner::NFQ, traj::AbstractTrajectory, ::AbstractEnv, ::PreEpisodeStage) end
function RLBase.update!(learner::NFQ, traj::AbstractTrajectory, env::AbstractEnv, ::PostEpisodeStage)
    isempty(traj) && return
    inds, batch = sample(learner.rng, traj, learner.sampler)
    update!(learner, batch, env)
end

function RLBase.update!(learner::NFQ, batch::NamedTuple{SARTS}, env::AbstractEnv)
    Q = learner.approximator
    γ = learner.γ
    loss_func = learner.loss_function
    as = action_space(env)
    las = length(as)

    (s, a, r, ss) = batch[[:state, :action, :reward, :next_state]]
    a = Float32.(a)
    s, a, r, ss = map(x->send_to_device(device(Q), x), (s, a, r, ss))
    for i = 1:learner.num_iterations
        # Make an input x samples x |action space| array -- Q --> samples x |action space| -- max --> samples
        G = r .+ γ .* (cat(repeat(ss, inner=(1, 1, las)), reshape(repeat(as, outer=(1, size(ss, 2))), (1, size(ss, 2), las)), dims=1) |> Q |> x -> maximum(x, dims=3) |> vec)
        for e = 1:learner.epochs
            Flux.train!((x, y) -> loss_func(Q(x), y), params(Q.model), [(vcat(s, transpose(a)), transpose(G))], Q.optimizer)
        end
    end
end
