using ReinforcementLearning

using Documenter
using Documenter.Writers.HTMLWriter
using Documenter.Utilities.DOM
using Documenter.Utilities.DOM: Tag, @tags

"filter concrete subtypes"
is_concrete_type_of(t) = x -> begin
    println(x)
    x isa Type && x <: t && !isabstracttype(x)
end

const top_nav = """
<div id="top" class="navbar-wrapper">
<nav class="navbar navbar-expand-lg  navbar-dark fixed-top" style="background-color: #1fd1f9; background-image: linear-gradient(315deg, #1fd1f9 0%, #b621fe 74%); " id="mainNav">
  <div class="container-md">
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarTogglerDemo01" aria-controls="navbarTogglerDemo01" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
    </button>
  <div class="collapse navbar-collapse" id="navbarTogglerDemo01">
    <span class="navbar-brand">
        <a class="navbar-brand" href="/">
          <!-- <img src="/assets/site/logo.svg" width="30" height="30" alt="logo" loading="lazy"> -->
          JuliaReinforcementLearning
        </a>
    </span>

    <ul class="navbar-nav ml-auto">
        <li class="nav-item">
        <a class="nav-link" href="/get_started/">Get Started</a>
        </li>
        <li class="nav-item">
        <a class="nav-link" href="/guide/">Guide</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="/contribute/">Contribute</a>
        </li>
        <li class="nav-item">
        <a class="nav-link" href="/blog/">Blog</a>
        </li>
        <li class="nav-item">
        <a class="nav-link" href="https://JuliaReinforcementLearning.github.io/ReinforcementLearning.jl/latest/">Doc</a>
        </li>
        <li class="nav-item">
        <a class="nav-link" href="https://github.com/JuliaReinforcementLearning">Github</a>
        </li>
    </ul>
  </div>
</nav>
</div>
"""

function HTMLWriter.render_html(
    ctx,
    navnode,
    head,
    sidebar,
    navbar,
    article,
    footer,
    scripts::Vector{DOM.Node} = DOM.Node[],
)
    @tags html body div script
    DOM.HTMLDocument(
        html[:lang=>"en"](
            head,
            body(
                Tag(Symbol("#RAW#"))(top_nav),
                div[".documenter-wrapper#documenter"](
                    sidebar,
                    div[".docs-main"](navbar, article, footer),
                    HTMLWriter.render_settings(ctx),
                ),
            ),
            scripts...,
        ),
    )
end

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
        canonical = "https://juliareinforcementlearning.github.io/ReinforcementLearning.jl/latest/",
        assets = [
            "assets/favicon.ico",
            "assets/custom.css",
            asset(
                "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
            ),
        ],
    ),
    sitename = "ReinforcementLearning.jl",
    linkcheck = !("skiplinks" in ARGS),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "RLBase" => "rl_base.md",
            "RLCore" => "rl_core.md",
            "RLEnvs" => "rl_envs.md",
            "RLZoo" => "rl_zoo.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/JuliaReinforcementLearning/ReinforcementLearning.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
