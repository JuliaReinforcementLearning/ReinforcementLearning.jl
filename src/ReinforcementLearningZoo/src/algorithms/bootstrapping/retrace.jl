export retrace
#Note: the speed of this operator may be improved by batching states and actions 
#to make single calls to Q, instead of one per batch sample. However this operator
#is no backpropagated through and its computation will typically represent a minor
#fraction of the runtime of a deep RL algorithm.

function retrace_operator(qnetwork, policy, batch, γ, λ)
    s = batch[:state] |> send_to_device(qnetwork)
    a = batch[:action] |> send_to_device(qnetwork)
    behavior_log_probs = batch[:action_log_problog_prob] |> send_to_device(qnetwork)
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
