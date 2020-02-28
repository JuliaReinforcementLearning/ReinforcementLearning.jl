using Documenter
using ReinforcementLearning

"filter concrete subtypes"
is_concrete_type_of(t) = x ->begin
    println(x)
    x isa Type && x <: t && !isabstracttype(x)
end

makedocs(
    modules = [ReinforcementLearning, ReinforcementLearningBase, ReinforcementLearningCore, ReinforcementLearningEnvironments, ReinforcementLearningZoo],
    format = Documenter.HTML(
        prettyurls = true,
        analytics = "UA-149861753-1",
        canonical = "https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/",
        assets = [
            "assets/favicon.ico",
            "assets/custom.css"
            ],
        ),
    sitename = "ReinforcementLearning.jl",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "Overview" => "overview.md",
        "A Quick Example" => "a_quick_example.md",
        "Manual" => [
            "RLBase" => "rl_base.md",
            "RLCore" => "rl_core.md",
            "RLEnvs" => "rl_envs.md",
            "RLZoo" => "rl_zoo.md",
        ],
        "Tips for Developers" => "tips_for_developers.md",
        "Experiments" => [
            "Play Atari Games with DQN" => "experiments/atari_dqn.md"
        ]],
)

deploydocs(
    repo = "github.com/JuliaReinforcementLearning/ReinforcementLearning.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)