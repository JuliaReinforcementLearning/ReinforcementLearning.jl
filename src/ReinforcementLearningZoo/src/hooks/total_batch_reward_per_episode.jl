
export TotalBatchRewardPerEpisode

#####
# TotalBatchRewardPerEpisode
#####
struct TotalBatchRewardPerEpisode{T,F} <: AbstractHook where {T<:Union{Val{true},Val{false}}, F<:Number}
    rewards::Vector{Vector{F}}
    reward::Vector{F}
    is_display_on_exit::Bool
end

Base.getindex(h::TotalBatchRewardPerEpisode) = h.rewards

"""
    TotalBatchRewardPerEpisode(batchsize::Int; is_display_on_exit=true)

Similar to [`TotalRewardPerEpisode`](@ref), but is specific to environments
which return a `Vector` of rewards (a typical case with `MultiThreadEnv`).
If `is_display_on_exit` is set to `true`, a ribbon plot will be shown to reflect
the mean and std of rewards.
"""
function TotalBatchRewardPerEpisode{F}(batchsize::Int; is_display_on_exit::Bool = true) where {F<:Number}
    TotalBatchRewardPerEpisode{is_display_on_exit, F}(
        [[] for _ = 1:batchsize],
        zeros(F, batchsize),
        is_display_on_exit,
    )
end

function TotalBatchRewardPerEpisode(batchsize::Int; is_display_on_exit::Bool = true)
    TotalBatchRewardPerEpisode{Float64}(batchsize; is_display_on_exit = is_display_on_exit)
end


function Base.push!(hook::TotalBatchRewardPerEpisode, 
    ::PostActStage,
    agent,
    env,
)
    hook.reward .+= reward(env)
    return
end

function Base.push!(hook::TotalBatchRewardPerEpisode, 
    ::PostActStage,
    agent::P,
    env::E,
    player::Symbol,
) where {P <: AbstractPolicy, E <: AbstractEnv}
    hook.reward .+= reward(env, player)
    return
end

function Base.push!(hook::TotalBatchRewardPerEpisode, ::PostEpisodeStage, agent, env)
    push!.(hook.rewards, copy(hook.reward))
    hook.reward .= 0
    return
end

function Base.show(io::IO, hook::TotalBatchRewardPerEpisode{true, F}) where {F<:Number}
    if sum(length(i) for i in hook.rewards) > 0
        n = minimum(map(length, hook.rewards))
        m = mean([@view(x[1:n]) for x in hook.rewards])
        s = std([@view(x[1:n]) for x in hook.rewards])
        p = lineplot(
            m,
            title = "Avg total reward per episode",
            xlabel = "Episode",
            ylabel = "Score",
        )
        lineplot!(p, m .- s)
        lineplot!(p, m .+ s)
        println(io, p)
    else
        println(io, typeof(hook))
    end
end

function Base.push!(hook::TotalBatchRewardPerEpisode{true, F}, 
    ::PostExperimentStage,
    agent,
    env,
) where {F<:Number}
    display(hook)
end

# Pass through as no need for multiplayer customization
function Base.push!(hook::TotalBatchRewardPerEpisode, 
    stage::Union{PostEpisodeStage, PostExperimentStage},
    agent,
    env,
    player::Symbol
)
    push!(hook,
        stage,
        agent,
        env,
    )
end
