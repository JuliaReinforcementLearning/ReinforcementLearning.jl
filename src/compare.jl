using Random: seed!
import DataFrames: DataFrame, groupby
import Colors: distinguishable_colors
using PGFPlotsX
export compare
export plotcomparison

"""
    compare(rlsetupcreators::Dict, N; callbackid = 1, verbose = false)

Run different setups in dictionary `rlsetupcreators` `N` times. The dictionary
has elements `"name" => createrlsetup`, where `createrlsetup` is a function that
has a single integer argument (id of the comparison; useful for saving 
intermediate results). For each run, `getvalue(rlsetup.callbacks[callbackid])`
gets entered as result in a DataFrame with columns "name", "result", "seed".
It is useful to specify the `callbackid` if the `createrlsetup` creates `RLSetup`s
with multiple callbacks.

# Example
```julia
using ReinforcementLearningEnvironmentDiscrete
env = MDP()
setup(learner) = RLSetup(learner, env, ConstantNumberSteps(10^4), callbacks = [EvaluationPerT(10^2, MeanReward())])
rlsetupcreators = Dict("sarsa" => (i) -> setup(Sarsa()), "smallbackups" => (i) -> setup(SmallBackups()))
result = compare(rlsetupcreators, 4)
plotcomparison(result)
```
"""
function compare(rlsetupcreators, N; callbackid = 1, verbose = false)
    res = @distributed(hcat, for t in 1:N
        seed = rand(1:typemax(UInt64)-1)
        tmp = []
        for (name, setupcreator) in rlsetupcreators
            if verbose
                @info("$(now()) \tStarting comparison $t, setup $name with seed $seed.")
            end
            seed!(seed)
            rlsetup = setupcreator(t)
            learn!(rlsetup)
            push!(tmp, [name, getvalue(rlsetup.callbacks[callbackid]), seed])
        end
        hcat(tmp...)
    end)
    DataFrame(name = res[1,:], result = res[2,:], seed = res[3,:])
end

"""
    plotcomparison(df; nmaxpergroup = 20, linestyles = [], 
                       showbest = true, axisoptions = @pgf {})

Plots results obtained with [`compare`](@ref) using [PGFPlotsX](https://github.com/KristofferC/PGFPlotsX.jl).
"""
function plotcomparison(df; nmaxpergroup = 20, linestyles = [], 
                        showbest = true, axisoptions = @pgf {})
    groups = groupby(df, :name)
    linestyles = linestyles == [] ? map(c -> @pgf({color = c}), 
                                        distinguishable_colors(length(groups))) : 
                                    linestyles
    plots = []
    legendentries = []
    isnumber = typeof(df[:result][1]) <: Number
    for (i, g) in enumerate(groups)
        if isnumber
            push!(plots, @pgf Plot({boxplot}, Table({y_index = 0}, 
                                                    Dict("res" => g[:result]))))
            push!(legendentries, g[:name][1])
        else
            m = mean(g[:result])
            push!(plots, @pgf Plot({thick, style = linestyles[i]}, 
                                   Coordinates(1:length(m), m)))
            push!(legendentries, g[:name][1])
            if showbest
                ma = g[:result][argmax(map(mean, g[:result]))]
                push!(plots, @pgf Plot({thick,dashed, style = linestyles[i]}, 
                                       Coordinates(1:length(ma), ma)))
                push!(legendentries, "")
            end
            for k in 1:min(nmaxpergroup, length(g[:result]))
                push!(plots, @pgf Plot({very_thin, style = linestyles[i], opacity = .3},
                                       Coordinates(1:length(g[:result][k]),
                                                   g[:result][k])))
                push!(legendentries, "")
            end
        end
    end
    if isnumber
        @pgf Axis({"boxplot/draw direction=y", 
                   xticklabels = legendentries,
                   xtick = collect(1:length(legendentries)),
                   axisoptions...}, plots...)
    else
        @pgf Axis({no_markers, "legend pos" = "outer north east", axisoptions...}, 
                  plots..., Legend(legendentries))
    end
end
