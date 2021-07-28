using ElasticArrays

Base.push!(a::ElasticArray, x) = append!(a, x)
Base.push!(a::ElasticArray{T,1}, x) where {T} = append!(a, [x])
Base.empty!(a::ElasticArray) = ElasticArrays.resize_lastdim!(a, 0)

function Base.pop!(a::ElasticArray)
    if length(a) > 0
        last_frame_inds = length(a.data)-a.kernel_length.divisor+1:length(a.data)
        d = reshape(view(a.data, last_frame_inds), a.kernel_size)
        ElasticArrays.resize!(a.data, length(a.data) - a.kernel_length.divisor)
        d
    else
        @error "can not pop! from an empty ElasticArray"
    end
end
