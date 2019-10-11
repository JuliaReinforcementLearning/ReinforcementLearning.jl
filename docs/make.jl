using Documenter
using ReinforcementLearning, ReinforcementLearningEnvironments

makedocs(
    modules = [ReinforcementLearning, ReinforcementLearningEnvironments],
    format = Documenter.HTML(
        prettyurls = true,
        canonical = "https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/",
        assets = ["assets/favicon.ico"],
        ),
    sitename = "ReinforcementLearning.jl",
    linkcheck = !("skiplinks" in ARGS),
    pages = [ "Home" => "index.md" ],
)

deploydocs(
    repo = "github.com/JuliaReinforcementLearning/ReinforcementLearning.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)