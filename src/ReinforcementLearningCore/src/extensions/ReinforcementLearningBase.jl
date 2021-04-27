using AbstractTrees

Base.show(io::IO, p::AbstractPolicy) =
    AbstractTrees.print_tree(io, StructTree(p), get(io, :max_depth, 10))

is_expand(::AbstractEnv) = false
