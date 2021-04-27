# inspired by
# - https://github.com/dalum/InlineExports.jl/blob/master/src/InlineExports.jl
# - https://github.com/QuantumBFS/YaoBase.jl/blob/master/src/utils/interface.jl

# TODO: use MLStyle.jl instead!

const API = []
const ENV_API = []
const MULTI_AGENT_ENV_API = []

macro api(ex)
    interfacem(__module__, __source__, ex, API)
end

macro env_api(ex)
    interfacem(__module__, __source__, ex, ENV_API)
end

macro multi_agent_env_api(ex)
    interfacem(__module__, __source__, ex, MULTI_AGENT_ENV_API)
end

function interfacem(__module__::Module, __source__::LineNumberNode, ex::Expr, store)
    name, is_body_missing = handle(ex)
    if name === nothing
        :(error("unknown expression"))
    else
        push!(store, name)
        if is_body_missing
            quote
                export $(esc(name))
                Core.@__doc__ $(esc(ex)) = error("method not implemented")
            end
        else
            quote
                export $(esc(name))
                Core.@__doc__ $(esc(ex))
            end
        end
    end
end

handle(ex) = extract_name(ex), is_body_missing(ex)

extract_name(::Any) = nothing
extract_name(x::Symbol) = x
extract_name(x::QuoteNode) = x.value
extract_name(ex::Expr) = extract_name(Val(ex.head), ex)
extract_name(::Val{:abstract}, ex) = extract_name(ex.args[1])
extract_name(::Val{:(=)}, ex) = extract_name(ex.args[1])
extract_name(::Val{:call}, ex) = extract_name(ex.args[1])
extract_name(::Val{:(::)}, ex) = extract_name(ex.args[2])
extract_name(::Val{:(<:)}, ex) = extract_name(ex.args[1])
extract_name(::Val{:curly}, ex) = extract_name(ex.args[1])
extract_name(::Val{:.}, ex) = extract_name(ex.args[2])
extract_name(::Val{:where}, ex) = extract_name(ex.args[1])
extract_name(::Val{:function}, ex) = extract_name(ex.args[1])
extract_name(::Val{:struct}, ex) = extract_name(ex.args[2])
extract_name(::Val{:const}, ex) = extract_name(ex.args[1])

is_body_missing(::Any) = false

is_body_missing(ex::Expr) = is_body_missing(Val(ex.head), ex)
is_body_missing(::Any, ::Any) = false
is_body_missing(::Val{:call}, ::Any) = true
