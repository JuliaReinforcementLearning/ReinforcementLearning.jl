using AbstractTrees
using Random
using ProgressMeter: Progress

const AT = AbstractTrees

struct StructTree{X}
    x::X
end

is_expand(x) = true
is_expand(::AbstractArray) = false
is_expand(::AbstractDict) = false
is_expand(::AbstractRNG) = false
is_expand(::Progress) = false
is_expand(::Function) = false
is_expand(::UnionAll) = false
is_expand(::DataType) = false

function AT.children(t::StructTree{X}) where {X}
    if is_expand(t.x)
        Tuple(f => StructTree(getfield(t.x, f)) for f in fieldnames(X))
    else
        ()
    end
end

AT.children(t::Pair{Symbol,<:StructTree}) = children(last(t))

AT.printnode(io::IO, t::StructTree{T}) where {T} = print(io, T.name)
AT.printnode(io::IO, t::StructTree{<:Union{Number,Symbol}}) = print(io, t.x)
AT.printnode(io::IO, t::StructTree{UnionAll}) = print(io, t.x)
AT.printnode(io::IO, t::StructTree{<:AbstractArray}) = summary(io, t.x)

function AT.printnode(io::IO, t::Pair{Symbol,<:StructTree})
    print(io, first(t), " => ")
    AT.printnode(io, last(t))
end

function AT.printnode(io::IO, t::StructTree{String})
    s = t.x
    i = findfirst('\n', s)
    if isnothing(i)
        if length(s) > 79
            print(io, "\"$(s[1:79])...\"")
        else
            print(io, "\"$s\"")
        end
    else
        if i > 79
            print(io, "\"$(s[1:79])...\"")
        else
            print(io, "\"$(s[1:i-1])...\"")
        end
    end
end

AT.printnode(io::IO, t::Pair{Symbol,<:StructTree{<:Tuple}}) = print(io, first(t))
