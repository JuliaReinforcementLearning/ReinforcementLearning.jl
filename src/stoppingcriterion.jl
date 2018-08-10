"""
    mutable struct ConstantNumberSteps
        T::Int64
        counter::Int64

Stops learning when the agent has taken 'T' actions.
"""
mutable struct ConstantNumberSteps
    N::Int64
    counter::Int64
end
"""
    ConstantNumberSteps(N) = ConstantNumberSteps(N, 0)
"""
ConstantNumberSteps(N) = ConstantNumberSteps(N, 0)
function isbreak!(criterion::ConstantNumberSteps, sraw, a, r, done)
    criterion.counter += 1
    if criterion.counter == criterion.N
        criterion.counter = 0
        return true
    end
    false
end
export ConstantNumberSteps

"""
    mutable struct ConstantNumberEpisodes
        N::Int64
        counter::Int64

Stops learning when the agent has finished 'N' episodes.
"""
mutable struct ConstantNumberEpisodes
    N::Int64
    counter::Int64
end
"""
        ConstantNumbeEpisodes(N) = ConstantNumberEpisodes(N, 0)
"""
ConstantNumberEpisodes(N) = ConstantNumberEpisodes(N, 0)
function isbreak!(criterion::ConstantNumberEpisodes, sraw, a, r, done)
    if done
        criterion.counter += 1
        if criterion.counter == criterion.N
            criterion.counter = 0
            return true
        end
    end
    false
end
export ConstantNumberEpisodes

