using MacroTools: @forward

#####
# Utils
#####

using IntervalSets

Random.rand(s::Interval) = rand(Random.GLOBAL_RNG, s)

function Random.rand(rng::AbstractRNG, s::Interval)
    r = rand(rng)
    
    # Check to prevent choosing an excluded endpoint
    # of (half-)open intervals
    while (r == 0.0) || (r == 1.0)
        r = rand(rng)
    end

    return r * (s.right - s.left) + s.left
end

#####
# WorldSpace
#####

#####
# ZeroTo
#####

export ZeroTo

"""
Similar to `Base.OneTo`. Useful when wrapping third-party environments.
"""
struct ZeroTo{T<:Integer} <: AbstractUnitRange{T}
    stop::T
    ZeroTo{T}(n) where {T<:Integer} = new(max(zero(T) - one(T), n))
end

ZeroTo(n::T) where {T<:Integer} = ZeroTo{T}(n)

Base.show(io::IO, r::ZeroTo) = print(io, "ZeroTo(", r.stop, ")")
Base.length(r::ZeroTo{T}) where {T} = T(r.stop + one(r.stop))
Base.first(r::ZeroTo{T}) where {T} = zero(r.stop)

function Base.getindex(v::ZeroTo{T}, i::Integer) where {T}
    Base.@_inline_meta
    @boundscheck ((i > 0) & (i <= v.stop+1)) || Base.throw_boundserror(v, i)
    convert(T, i-1)
end

#####
# Space
#####

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
            "<li> <a href=\"https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.$(nameof(env))-Tuple{}\"> $(nameof(env)) </a></li>",
        )
    end
    print(io, "</ol>")
end
