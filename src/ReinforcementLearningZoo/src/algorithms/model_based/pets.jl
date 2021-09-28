include("cem_optimizer.jl")

export PETSPolicy

mutable struct PETSPolicy{
    O<:CEMTrajectoryOptimizer,
    E<:NeuralNetworkApproximator,
    P<:AbstractPolicy,
    R<:AbstractRNG,
} <: AbstractPolicy
    optimizer::O
    ensamble::Vector{E}
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_freq::Int
    update_step::Int
    rng::R
end

function PETSPolicy(;
    optimizer,
    ensamble,
    batch_size=64,
    start_steps=100,
    start_policy,
    update_after=100,
    update_freq=100,
    rng = Random.GLOBAL_RNG,
)
    PETSPolicy(
        optimizer,
        ensamble,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_freq,
        0,
        rng
    )
end

function (p::PETSPolicy)(env)
    p.update_step += 1

    if p.update_step <= p.start_steps
        a = p.start_policy(env)
        return a
    else
        # TODO is it bad to use closure here?
        function trajectory_eval(action_sequence)  
            s = state(env)
            # Do this for all sequences at the same time later
            rtot = 0f0
            for i in 1:size(action_sequence, 2)
                ens_out = [m(vcat(s, action_sequence[:, i]); is_sampling=true) for m in p.ensamble]
                ens_mean = mean(ens_out)
                rtot += ens_mean[end]
                s = ens_mean[1:end-1]
            end
            return rtot
        end
        p.optimizer(trajectory_eval)
    end
end

function RLBase.update!(
    p::PETSPolicy,
    traj::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    length(traj) > p.update_after || return
    p.update_step % p.update_freq == 0 || return
    inds, batch = sample(p.rng, traj, BatchSampler{SARTS}(p.batch_size))
    update!(p, batch)
end

function RLBase.update!(p::PETSPolicy, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(p.ensamble[1]), batch) # TODO merge ensamble to type?

    state_action = vcat(s, a)
    nstate_reward = vcat(s′, Flux.unsqueeze(r, 1))

    # Train each model
    for m in p.ensamble
        grad = gradient(Flux.params(m)) do
            μ, logσ = m(p.rng, state_action) 
            mean(((nstate_reward .- μ) ./ exp.(logσ)) .^ 2 ./ 2 .+ logσ)
        end
        update!(m, grad)
    end
end