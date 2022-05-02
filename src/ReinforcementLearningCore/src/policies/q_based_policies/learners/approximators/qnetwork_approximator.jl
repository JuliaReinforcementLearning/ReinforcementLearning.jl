export AbstractQNetwork, QNetwork, TwinQNetwork

#####################
#QNetwork
#####################

abstract type AbstractQNetwork <: AbstractApproximator end

#An algorithm that uses a QNetwork can call this function to update it. This will dispatch depending on the type of QNetwork used (twin or not)
function update_qnetwork!(p::AbstractPolicy, t::AbstractTrajectory)
    update!(p, t, p.qnetwork)
end

#QNetwork is a combination of two identical NNs, one being a target of bootstrapping. tau is the polyak averaging update weight or a boolean, if false, no target network is used (target will point to the main network). 
mutable struct QNetwork{Q <: NeuralNetworkApproximator, T<:Union{Float32, Bool}, F} <: AbstractQNetwork
    qnetwork::Q
    target_qnetwork::Q
    τ::T
    loss::F
end

@functor QNetwork

Flux.trainable(q::QNetwork) = (q.qnetwork,)

QNetwork(nn; τ = 0.01f0, target = false, loss = Flux.mse) = QNetwork(nn, target ? deepcopy(nn) : nn, target ? τ : false, loss)

(qnetwork::QNetwork)(states, actions) = qnetwork.qnetwork([states; actions]) #how do we handle image states? Can't concatenate with actions.

#A QNetwork is updated given `states, actions` to minimise `mse(targets, Q(states,actions)`
function update!(qnetwork::QNetwork, states, actions, targets)
    τ = qnetwork.τ
    ps = Flux.params(qnetwork)
    gs = gradient(ps) do 
        qs = qnetwork(states, actions) |> vec
        qnetwork.loss(targets, qs)
    end
    if any(x -> !isnothing(x) && any(y -> isnan(y) || isinf(y), x), gs)
        error("Gradient of QNetwork contains NaN of Inf")
    end
    Flux.update!(qnetwork.optimizer, ps, gs)

    if τ != false
        for (dest, src) in zip(Flux.params(qnetwork.target_qnetwork), Flux.params(qnetwork.qnetwork))
            dest .= (1 - τ) .* dest .+ τ .* src
        end
    end
end

#called by the function above, or directly in-algorithm
function update!(p, t, qnetwork::QNetwork)
    inds, batch = p.batch_sampler(t) #use the policy's batch_sampler to get a batch
    y = q_targets(p, t, qnetwork, batch) #compute targets for qnetwork, can be specialised by overloading on a trajectory or policy type
    update!(qnetwork, batch.state, batch.action, y)
end

#TD target using the target network. May be overloaded with other trajectories or batch traces
function q_targets(p::AbstractPolicy, ::AbstractTrajectory, qnetwork::QNetwork, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = batch.state, batch.action, batch.reward, batch.terminal, batch.next_state
    a′ = p.policy(p.rng, s′; is_sampling=true, is_return_log_prob=false)
    q′_input = vcat(s′, a′)
    q′ = qnetwork.target_qnetwork(q′_input)
    return r .+ p.γ .* (1 .- t) .* vec(q′)
end

#####################
#TwinQNetwork
#####################

mutable struct TwinQNetwork{Q<:QNetwork} <: AbstractQNetwork
    qnetwork1::Q 
    qnetwork2::Q
end

@functor TwinQNetwork

(qnetwork::TwinQNetwork)(states, actions) = qnetwork.qnetwork1(states, actions) #how do we handle image states? Can't concatenate with actions.

#when we have twins, we must update each one
function update!(p, t, twinqnetwork::TwinQNetwork)
    inds, batch = p.batch_sampler(t)
    y = q_targets(p, t, twinqnetwork.qnetwork1, twinqnetwork.qnetwork2, batch...)
    s = batch.state
    a = batch.action
    update!(twinqnetwork.qnetwork1, y, s, a)

    inds, batch = p.batch_sampler(t)
    y = q_targets(p, t, twinqnetwork.qnetwork2, twinqnetwork.qnetwork1, batch...)
    s = batch.state
    a = batch.action
    update!(twinqnetwork.qnetwork2, y, s, a)
end

#TD target of qnetwork using the twin target networks. May be overloaded with other trajectories or batch traces
function q_targets(p::AbstractPolicy, ::AbstractTrajectory, qnetwork, twinqnetwork, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = send_to_device(device(qnetwork.qnetwork), batch)
    γ = p.γ

    a′ = p.policy(p.rng, s′; is_sampling=true, is_return_log_prob=false)
    q′_input = vcat(s′, a′)
    q′ = min.(qnetwork.target_qnetwork(q′_input), twinqnetwork.target_qnetwork(q′_input))

    y = r .+ γ .* (1 .- t) .* vec(q′)
end