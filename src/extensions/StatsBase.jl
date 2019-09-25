import StatsBase: countmap

"extend the countmap in StatsBase to support general iterator"
function countmap(iter)
    res = Dict{eltype(iter),Int}()
    for x in iter
        if haskey(res, x)
            res[x] += 1
        else
            res[x] = 1
        end
    end
    res
end