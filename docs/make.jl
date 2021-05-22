using ReinforcementLearning
using Documenter
using Markdown
using DemoCards

open(joinpath(@__DIR__, "..", "README.md"), "r") do f_src
    open(joinpath(@__DIR__, "src", "index.md"), "w") do f_dest
        s_dest = read(f_src, String)
        s_dest = replace(s_dest, "<!-- ```@raw html -->" => "```@raw html")
        s_dest = replace(s_dest, "<!-- ``` -->" => "```")
        write(f_dest, s_dest)
    end
end

experiments, postprocess_cb, experiments_assets = makedemos("experiments")

assets = [
    "assets/favicon.ico",
    "assets/custom.css",
    experiments_assets
]

makedocs(
    modules = [
        ReinforcementLearning,
        ReinforcementLearningBase,
        ReinforcementLearningCore,
        ReinforcementLearningEnvironments,
        ReinforcementLearningZoo,
    ],
    format = Documenter.HTML(
        prettyurls = true,
        analytics = "UA-149861753-1",
        assets = assets,
    ),
    sitename = "ReinforcementLearning.jl",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Guides" => [
            "How to write a customized environment?" => "How_to_write_a_customized_environment.md",
            "How to implement a new algorithm?" => "How_to_implement_a_new_algorithm.md",
            "How to use hooks?" => "How_to_use_hooks.md",
            "Which algorithm should I use?" => "Which_algorithm_should_I_use.md",
        ],
        "FAQ" => "FAQ.md",
        experiments,
        "Tips for Developers" => "tips.md",
        "Manual" => [
            "RLBase" => "rlbase.md",
            "RLCore" => "rlcore.md",
            "RLEnvs" => "rlenvs.md",
            "RLZoo" => "rlzoo.md",
        ],
    ],
)

postprocess_cb()
