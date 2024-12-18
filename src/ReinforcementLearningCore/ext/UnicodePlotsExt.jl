module UnicodePlotsExt
    using ReinforcementLearningCore
    using UnicodePlots: lineplot, lineplot!

    function Base.show(io::IO, hook::TotalRewardPerEpisode{true, F}) where {F<:Number}
        if length(hook.rewards) > 0
            println(io, lineplot(
                hook.rewards,
                title="Total reward per episode",
                xlabel="Episode",
                ylabel="Score",
            ))
        else
            println(io, typeof(hook))
        end
        return
    end
end
