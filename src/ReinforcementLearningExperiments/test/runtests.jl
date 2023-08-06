using UUIDs
using Preferences

if Sys.isapple()
    flux_uuid = UUID("587475ba-b771-5e3f-ad9e-33799f191a9c")
    set_preferences!(flux_uuid, "gpu_backend" => "Metal")

    using Metal
else
    using CUDA, cuDNN
    CUDA.allowscalar(false)
end

using ReinforcementLearningExperiments
using Flux
println("Flux.GPU_BACKEND = $(Flux.GPU_BACKEND)")
import ReinforcementLearningCore: RLCore

using StatsBase
using JET
using BenchmarkTools

# ex = E`JuliaRL_BasicDQN_CartPole`
ex = E`JuliaRL_DQN_CartPole`
stop_condition = StopAfterStep(10_000, is_show_progress=true)
RLCore._run(ex.policy, ex.env, stop_condition, ex.hook, ResetAtTerminal())

batch = StatsBase.sample(ex.policy.trajectory)
learner = ex.policy.policy.learner

optimiser_state = learner.approximator.optimiser_state
A = learner.approximator
Q = A.model.source
Qₜ = A.model.target

γ = learner.γ
loss_func = learner.loss_func
n = learner.n

s, s_next, a, r, t = map(x -> batch[x], SS′ART) |> Flux.gpu
a = CartesianIndex.(a, 1:length(a))

q_next = learner.is_enable_double_DQN ? Q(s_next) : Qₜ(s_next)

if haskey(batch, :next_legal_actions_mask)
    q_next .+= ifelse.(batch[:next_legal_actions_mask], 0.0f0, typemin(Float32))
end

q_next_action = learner.is_enable_double_DQN ? Qₜ(s_next)[dropdims(argmax(q_next, dims=1), dims=1)] : dropdims(maximum(q_next; dims=1), dims=1)

R = r .+ γ^n .* (1 .- t) .* q_next_action

grads = gradient(Q) do Q
    qₐ = Q(s)[a]
    loss = loss_func(R, qₐ)
    ignore_derivatives() do
        learner.loss[1] = loss
    end
    loss
end |> Flux.cpu

# Optimization step
Flux.update!(optimiser_state, Flux.cpu(Q), grads[1])
