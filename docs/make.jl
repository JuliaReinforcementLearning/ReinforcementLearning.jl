using Documenter
using ReinforcementLearning, ReinforcementLearningEnvironments

makedocs(
    modules = [ReinforcementLearning, ReinforcementLearningEnvironments],
    format = Documenter.HTML(
        prettyurls = true,
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
        "A Quick Example" => "a_quick_example.md",
        "Overview" => "overview.md",
        "Manual" => [
            "Core" => "core.md",
            "Components" => [
                "Agents" => "components/agents.md",
                "Buffers" => "components/buffers.md",
                "Policies" => "components/policies.md",
                "Learners" => "components/learners.md",
                "Approximators" => "components/approximators.md",
                "Action Selectors" => "components/action_selectors.md",
                "Environment Models" => "components/environment_models.md"
            ],
            "Utils" => "utils.md"
        ],
        "Tips for Developers" => "tips_for_developers.md"],
)

deploydocs(
    repo = "github.com/JuliaReinforcementLearning/ReinforcementLearning.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)