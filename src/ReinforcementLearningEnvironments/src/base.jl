using MacroTools: @forward

#####
# Utils
#####

using IntervalSets

Random.rand(s::Union{Interval, Array{<:Interval}}) = rand(Random.GLOBAL_RNG, s)

function Random.rand(rng::AbstractRNG, s::Interval)
    rand(rng) * (s.right - s.left) + s.left
end

#####
# WorldSpace
#####

export WorldSpace

"""
In some cases, we may not be interested in the action/state space.
One can return `WorldSpace()` to keep the interface consistent.
"""
struct WorldSpace{T} end

WorldSpace() = WorldSpace{Any}()

Base.in(x, ::WorldSpace{T}) where T = x isa T

#####
# ZeroTo
#####

export ZeroTo

"""
Similar to `Base.OneTo`. Useful when wrapping third-party environments.
"""
struct ZeroTo{T<:Integer} <: AbstractUnitRange{T}
    stop::T
    ZeroTo{T}(n) where {T<:Integer} = new(max(zero(T)-one(T),n))
end

ZeroTo(n::T) where {T<:Integer} = ZeroTo{T}(n)

Base.show(io::IO, r::ZeroTo) = print(io, "ZeroTo(", r.stop, ")")
Base.length(r::ZeroTo{T}) where T = T(r.stop + one(r.stop))
Base.first(r::ZeroTo{T}) where T = zero(r.stop)

function getindex(v::ZeroTo{T}, i::Integer) where T
    Base.@_inline_meta
    @boundscheck ((i >= 0) & (i <= v.stop)) || throw_boundserror(v, i)
    convert(T, i)
end

#####
# Space
#####

export Space

"""
A wrapper to treat each element as a sub-space which supports:

- `Base.in`
- `Random.rand`
"""
struct Space{T}
    s::T
end

@forward Space.s Base.getindex, Base.setindex!, Base.size, Base.length

Base.similar(s::Space, args...) = Space(similar(s.s, args...))

Random.rand(s::Space) = rand(Random.GLOBAL_RNG, s)

Random.rand(rng::AbstractRNG, s::Space) = map(s.s) do x
    rand(rng, x)
end

Random.rand(rng::AbstractRNG, s::Space{<:Dict}) = Dict(k=>rand(rng,v) for (k,v) in s.s)

function Base.in(X, S::Space)
    if length(X) == length(S.s)
        for (x,s) in zip(X, S.s)
            if x ∉ s
                return false
            end
        end
        return true
    else
        return false
    end
end

function Base.in(X::Dict, S::Space{<:Dict})
    if keys(X) == keys(S.s)
        for k in keys(X)
            if X[k] ∉ S.s[k]
                return false
            end
        end
        return true
    else
        return false
    end
end

#####
# Generate README
#####

gen_traits_table(envs) = gen_traits_table(stdout, envs)

function gen_traits_table(io, envs)
    trait_dict = Dict()
    for f in RLBase.env_traits()
        for env in envs
            if !haskey(trait_dict, f)
                trait_dict[f] = Set()
            end
            t = f(env)
            if f == StateStyle
                if t isa Tuple
                    for x in t
                        push!(trait_dict[f], nameof(typeof(x)))
                    end
                else
                    push!(trait_dict[f], nameof(typeof(t)))
                end
            else
                push!(trait_dict[f], nameof(typeof(t)))
            end
        end
    end

    println(io, "<table>")

    print(io, "<th colspan=\"2\">Traits</th>")
    for i in 1:length(envs)
        print(io, "<th> $(i) </th>")
    end

    for k in sort(collect(keys(trait_dict)), by = nameof)
        vs = trait_dict[k]
        print(io, "<tr> <th rowspan=\"$(length(vs))\"> $(nameof(k)) </th>")
        for (i, v) in enumerate(vs)
            if i != 1
                print(io, "<tr> ")
            end
            print(io, "<th> $(v) </th>")
            for env in envs
                if k == StateStyle && k(env) isa Tuple
                    ss = k(env)
                    if v in map(x -> nameof(typeof(x)), ss)
                        print(io, "<td> ✔ </td>")
                    else
                        print(io, "<td> </td> ")
                    end
                else
                    if nameof(typeof(k(env))) == v
                        print(io, "<td> ✔ </td>")
                    else
                        print(io, "<td> </td> ")
                    end
                end
            end
            println(io, "</tr>")
        end
    end

    println(io, "</table>")

    print(io, "<ol>")
    for env in envs
        println(
            io,
            "<li> <a href=\"https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.$(nameof(env))-Tuple{}\"> $(nameof(env)) </a></li>",
        )
    end
    print(io, "</ol>")
end
