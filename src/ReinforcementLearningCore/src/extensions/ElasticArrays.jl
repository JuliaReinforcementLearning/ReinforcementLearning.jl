using ElasticArrays

Base.push!(a::ElasticArray, x) = append!(a, x)
Base.empty!(a::ElasticArray) = ElasticArrays.resize_lastdim!(A, 0)

function Base.pop!(a::ElasticArray)
    # ??? Is it safe to do so?
    last_frame = selectdim(a, ndims(a), size(a, ndims(a)))
    ElasticArrays.resize_lastdim!(A, size(a, ndims(a))-1)
    last_frame
end