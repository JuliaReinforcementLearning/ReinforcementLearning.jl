include("cem_optimizer.jl")

export PETSPolicy

mutable struct PETSPolicy{
    O<:CEMTrajectoryOptimizer,
    E<:NeuralNetworkApproximator,
    P<:AbstractPolicy,
    R<:AbstractRNG,
} <: AbstractPolicy

    # Model and optimizer
    optimizer::O
    ensamble::Vector{E}

    # Params
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_freq::Int
    update_step::Int

    # Settings
    predict_reward::Bool

    # Rng
    rng::R
end

function PETSPolicy(;
    optimizer,
    ensamble,
    batch_size=256,
    start_steps=200,
    start_policy,
    update_after=200,
    update_freq=50,
    predict_reward=false,
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
        predict_reward,
        rng
    )
end

function (p::PETSPolicy)(env)
    p.update_step += 1

    if p.update_step <= p.start_steps
        a = p.start_policy(env)
        return a
    else
        # TODO: This creates a closure? Remember there was something undesirable with closures, but not what...
        p.optimizer() do action_sequence
            s = state(env)
            # TODO: Do this for all sequences at the same time later
            rtot = 0f0
            for i in 1:size(action_sequence, 2)
                # TODO: here we use the uncertainty by sampling, but is this really correct? Need to look in to the algorithm more.
                ens_out = [m(vcat(s, action_sequence[:, i]); is_sampling=true) for m in p.ensamble]
                ens_mean = mean(ens_out)
                if p.predict_reward # TODO: Seems like this would be a type instability, probably should be solved in some other way
                    rtot += ens_mean[end]
                    s = s + ens_mean[1:end-1]
                else
                    s = s + ens_mean
                    # TODO: this is not nice, forces users to have rewardoverridden env with specific pattern
                    # or an env that is specifically designed for this
                    rtot += reward(env; action=action_sequence[:, i], nstate=s)
                end
            end
            return rtot
        end
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
    target = p.predict_reward ? vcat(s′ - s, Flux.unsqueeze(r, 1)) : s′ - s

    # Train each model
    for m in p.ensamble
        grad = gradient(Flux.params(m)) do
            μ, logσ = m(p.rng, state_action) 
            mean(((target .- μ) ./ exp.(logσ)) .^ 2 ./ 2 .+ logσ)
        end
        update!(m, grad)
    end
end