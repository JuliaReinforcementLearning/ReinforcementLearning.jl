using ElasticArrays

Base.push!(a::ElasticArray, x) = append!(a, x)
Base.empty!(a::ElasticArray) = ElasticArrays.resize_lastdim!(a, 0)

function Base.pop!(a::ElasticArray)
    last_frame = select_last_frame(a) |> copy  # !!! ensure that we will not access invalid data
    ElasticArrays.resize!(a.data, length(a.data) - a.kernel_length.divisor)
    last_frame
end
