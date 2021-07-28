@def title = "Call for Contributions"
@def description = "Reinforcement learning has undergone tremendous progress in recent years. The growing number of algorithms begets the need for comprehensive tools and implementations. Thus we call for all kinds of contributions from the community, including bug reports, feature proposals and implementations of new algorithms or reinforcement learning environments. This page is used to track what we hope to be added in the next few years."
@def is_enable_toc = false
@def has_code = false
@def has_math = false
@def bibliography = "bibliography.bib"

@def front_matter = """
    {
        "authors": [
            {
                "author":"Jun Tian",
                "authorURL":"https://github.com/findmyway",
                "affiliation":"",
                "affiliationURL":""
            }
        ],
        "publishedDate":"$(now())",
        "citationText":"Jun Tian, $(Dates.format(now(), "Y"))"
    }"""

@def appendix = """
    ### Corrections
    If you see mistakes or want to suggest changes, please [create an issue](https://github.com/JuliaReinforcementLearning/JuliaReinforcementLearning.github.io/issues) on the source repository.
    """

## Environments

Many reinforcement learning environments in the Python world are not available in Julia yet. Though we can leverage [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) to interact with them, the overheads usually make this approach unacceptable. Following are some experiments we'd like to have, either by wrapping the underlying C/C++ libraries or rewriting in Julia from scratch.

- [bullet3](https://github.com/bulletphysics/bullet3). There're some discussions about how to write a wrapper [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl/issues/15). ⭐⭐⭐
- [rlcard](https://github.com/datamllab/rlcard). It is suggested to be rewritten in Julia.⭐
- [boxoban-levels](https://github.com/deepmind/boxoban-levels) and [mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban). ⭐
- [gym-minigrid](https://github.com/maximecb/gym-minigrid). Need a great design and rewritten in Julia.⭐
- [procgen](https://github.com/openai/procgen). Environments in Procgen are written in C. One key feature is that Procgen environments are randomized.⭐⭐
- [MAgent](https://github.com/PettingZoo-Team/MAgent). ⭐⭐

\aside{Here ⭐ means the difficulty.}

Beside writing environments, it would be great if a unified wrapper is also provided in [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl).

## Algorithms

Note that each algorithm is suggested to provide at least one reproducible experiment.

### Q-Learning Related

- QRDQN\dcite{dabney2018distributional}.
- FQF\dcite{yang2019fully}. This is the new SOTA among distributional methods.

### Policy Gradient

- Soft Actor-Critic\dcite{haarnoja2018soft}
- Twin Delayed DDPG\dcite{dankwa2019twin}

### Model Based

- MuZero\dcite{schrittwieser2019mastering}

### Counterfactual Regret

- Counterfactual multi-agent policy gradients\dcite{foerster2018counterfactual}
- Deep CFR\dcite{brown2019deep}

### Monte Carlo Tree Search

## Infrastructures

### Visualization

- Integration with [Dashboards.jl](https://github.com/waralex/Dashboards.jl).
- Create custom recipes in [Maike.jl](http://makie.juliaplots.org/).

### Distributed Computing

### Other Backends

- [Torch.jl](https://github.com/FluxML/Torch.jl)