export retrace
#Note: the speed of this operator may be improved by batching states and actions 
#to make single calls to Q, instead of one per batch sample. However this operator
#is no backpropagated through and its computation will typically represent a minor
#fraction of the runtime of a deep RL algorithm.

function retrace_operator(qnetwork, policy, batch, γ, λ)
    s = batch[:state] |> send_to_device(qnetwork)
    a = batch[:action] |> send_to_device(qnetwork)
    behavior_log_probs = batch[:log_prob] |> send_to_device(qnetwork)
    r = batch[:reward] |> send_to_device(qnetwork)
    t = last.(batch[:terminal]) |> send_to_device(qnetwork)
    ns = batch[:next_state] |> send_to_device(qnetwork)
    na = map(ns) do ns
        policy(ns, is_sampling = true, is_return_log_prob = false)
    end
    states = map(s,ns) do s, ns #concatenates all states, including the last state to compute deltas with the target Q
        cat(s,last(eachslice(ns, dims = ndims(ns))),dims=ndims(s))
    end
    actions = map(a, na) do a, na 
         cat(a,last(eachslice(na, dims = ndims(na))),dims=ndims(a))
    end

    current_log_probs = map(s,a) do s, a
        policy(s,a)
    end
   
    traces = map(current_log_probs, behavior_log_probs) do p,m
        @. λ*min(1, exp(p - m))
    end
    is_ratios = cumprod.(traces) #batchsized vector [[c1,c2,...,ct],[c1,c2,...,ct],...]
    
    Qp = target(qnetwork)

    δs = map(states, actions, r, t) do s, a, r, t
        q = vec(Qp(vcat(s,a)))
        q[end] *= t
        r .+ q[2:end].*γ .- q[1:end-1]
    end

    ops = map(is_ratios, δs) do ratios, deltas
        γs = γ .^ (0:(length(deltas)-1))
        sum(γs .* ratios .* deltas)
    end

    return ops #batchsize vector of operator
end 

batch = (state= [[1 2 3], [10 11 12]], 
        action = [[1 2 3],[10 11 12]], 
        log_prob = [log.([0.2,0.2,0.2]), log.([0.1,0.1,0.1])],
        reward = [[1f0,2f0,3f0],[10f0,11f0,12f0]], 
        terminal= [[0,0,1], [0,0,0]], 
        next_state = [[2 3 4],[11 12 13]])

current_log_probs = [log.([0.1/2,0.1/3,0.1/4]) for i in 1:2]
policy(x; is_sampling = true, is_return_log_prob = false) = identity(x)
policy(s,a) = current_log_probs[2]
qnetwork(x, args...) = x[1, :]
λ, γ = 0.9, 0.99
target(Qp) = Qp
send_to_device(x) = identity
retrace_operator(qnetwork, policy, batch, γ, λ)


#calculer à la main pour batch[2]
1*0.9*1/4*(1+0.99*2-1) + 0.99*0.9^2*1/4*1/6*(2+0.99*3-2) + 0.99^2*0.9^3*1/4*1/6*1/8*(3+0.99*4*1-3)
1*0.9*0.5*(10+0.99*11-10) + 0.99*0.9^2*0.5*1/3*(11+0.99*12-11) + 0.99^2*0.9^3*0.5*0.33*0.25*(12+0.99*13*0-12)
