export NoTraces, updatetrace!

"""
    struct NoTraces <: AbstractTraces

No eligibility traces, i.e. ``e(a, s) = 1`` for current action ``a`` and state
``s`` and zero otherwise.
"""
struct NoTraces end

# TODO: combine in one struct with additional field replacing::Bool
for kind in (:ReplacingTraces, :AccumulatingTraces)
    @eval (struct $kind{Tt} 
                λ::Float64
                γλ::Float64
                trace::Tt
                minimaltracevalue::Float64
            end;
            export $kind;
            function $kind(ns, na, λ::Float64, γ::Float64; 
                           minimaltracevalue = 1e-12,
                           trace = sparse([], [], Float64[], na, ns))
                $kind(λ, γ*λ, trace, minimaltracevalue)
            end)
end
@doc """
    struct ReplacingTraces <: AbstractTraces
        λ::Float64
        γλ::Float64
        trace::Array{Float64, 2}
        minimaltracevalue::Float64

Decaying traces with factor γλ. 

Traces are updated according to ``e(a, s) ←  1`` for the current action-state
pair and ``e(a, s) ←  γλ e(a, s)`` for all other pairs unless
``e(a, s) < `` `minimaltracevalue` where the trace is set to 0 
(for computational efficiency).
""" ReplacingTraces
@doc """
    ReplacingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)
""" ReplacingTraces()
@doc """
    struct AccumulatingTraces <: AbstractTraces
        λ::Float64
        γλ::Float64
        trace::Array{Float64, 2}
        minimaltracevalue::Float64

Decaying traces with factor γλ. 

Traces are updated according to ``e(a, s) ←  1 + e(a, s)`` for the current action-state
pair and ``e(a, s) ←  γλ e(a, s)`` for all other pairs unless
``e(a, s) < `` `minimaltracevalue` where the trace is set to 0 
(for computational efficiency).
""" AccumulatingTraces
@doc """
    AccumulatingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)
""" AccumulatingTraces()

function increasetrace!(traces::ReplacingTraces, state::Int, action)
    traces.trace[action, state] = 1.
end
function increasetrace!(traces::ReplacingTraces, state::Vector, action)
    @inbounds for i in findall(x -> x != 0, state)
        traces.trace[action, i] = state[i]
    end
end
function increasetrace!(traces::ReplacingTraces, state::SubArray{T, 1} where T, action)
    @inbounds for i in findall(x -> x != 0, state)
        traces.trace[action, i] = state[i]
    end
end
function increasetrace!(traces::ReplacingTraces, state::SparseVector, action)
    @inbounds for i in 1:length(state.nzind)
        traces.trace[action, state.nzind[i]] = state.nzval[i]
    end
end
function increasetrace!(traces::AccumulatingTraces, state::Int, action)
    if traces.trace[action, state] == 0
        traces.trace[action, state] = 1.
    else
        traces.trace[action, state] += 1.
    end
end
function increasetrace!(traces::AccumulatingTraces, state::Vector, action)
    @inbounds for i in findall(x -> x != 0, state)
        traces.trace[action, i] += state[i]
    end
end
function increasetrace!(traces::AccumulatingTraces, state::SparseVector, action)
    @inbounds for i in 1:length(state.nzind)
        traces.trace[action, state.nzind[i]] += state.nzval[i]
    end
end


discounttraces!(t) = discounttraces!(t.trace, t.γλ, t.minimaltracevalue)
@inline function discounttraces!(trace::SparseMatrixCSC, γλ, minimaltracevalue)
    x = trace.nzval
    @simd for i in 1:length(x)
        @inbounds if x[i] <= minimaltracevalue
                     x[i] = 0.
                  else
                     x[i] *= γλ
                  end
    end
    if rand() < .005
        dropzeros!(trace)
    end
end
@inline discounttraces!(trace, γλ, minimaltracevalue) = rmul!(trace, γλ)
@inline resettraces!(traces) = resettrace!(traces.trace)
@inline resettrace!(trace) = rmul!(trace, 0.)
@inline function resettrace!(trace::SparseMatrixCSC)
    rmul!(trace.nzval, 0.)
    dropzeros!(trace)
end

function updatetraceandparams!(traces, params, factor)
    updatetraceandparams!(traces.trace, params, factor)
    discounttraces!(traces)
end

@inline function updatetraceandparams!(s::SparseMatrixCSC, params, factor)
    c = 1
    @inbounds for k in 1:length(s.nzval)
        while s.colptr[c+1] - 1 < k || s.colptr[c] == s.colptr[c+1]; c += 1; end
        params[s.rowval[k], c] += factor * s.nzval[k]
    end
end
@inline updatetraceandparams!(t::AbstractArray, params, factor) = 
    axpy!(factor, t, params)


function updatetrace!(traces, state, action)
    discounttraces!(traces)
    increasetrace!(traces, state, action)
end
