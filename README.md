<div align="center">
  <p>
  <img src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/master/docs/src/assets/logo.svg?sanitize=true" width="320px">
  </p>

  <p>
  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/actions?query=workflow%3ACI"><img src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/workflows/CI/badge.svg"></a>
  <a href="https://juliahub.com/ui/Packages/ReinforcementLearning/6l2TO"><img src="https://juliahub.com/docs/ReinforcementLearning/pkgeval.svg"></a>
  <a href="https://juliahub.com/ui/Packages/ReinforcementLearning/6l2TO"><img src="https://juliahub.com/docs/ReinforcementLearning/version.svg"></a>
  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/LICENSE.md"><img src="http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat"></a>
  <a href="https://julialang.org/slack/"><img src=https://img.shields.io/badge/Chat%20on%20Slack-%23reinforcement--learnin-ff69b4"></a>
  <a href="https://github.com/SciML/ColPrac"><img src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet"></a>
  </p>

</div>

[**ReinforcementLearning.jl**](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl),
as the name says, is a package for reinforcement learning research in Julia.

Our design principles are:

- **Reusability and extensibility**: Provide elaborately designed components and
  interfaces to help users implement new algorithms.
- **Easy experimentation**: Make it easy for new users to run benchmark
  experiments, compare different algorithms, evaluate and diagnose agents.
- **Reproducibility**: Facilitate reproducibility from traditional tabular
  methods to modern deep reinforcement learning algorithms.

## Get Started

```julia
julia> ] add ReinforcementLearning

julia> using ReinforcementLearning

julia> run(E`JuliaRL_BasicDQN_CartPole`)
```

Check out the [Get Started](https://juliareinforcementlearning.org/get_started/) page for more detailed explanation!

## Project Structure

`ReinforcementLearning.jl` itself is just a wrapper around several other packages inside the [JuliaReinforcementLearning](https://github.com/JuliaReinforcementLearning) org. The relationship between different packages is described below:

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  ReinforcementLearning.jl                                                         |
|                                                                                   |
|      +------------------------------+                                             |
|      | ReinforcementLearningBase.jl |                                             |
|      +----|-------------------------+                                             |
|           |                                                                       |
|           |     +--------------------------------------+                          |
|           +---->+ ReinforcementLearningEnvironments.jl |                          |
|           |     +--------------------------------------+                          |
|           |                                                                       |
|           |     +------------------------------+                                  |
|           +---->+ ReinforcementLearningCore.jl |                                  |
|                 +----|-------------------------+                                  |
|                      |                                                            |
|                      |     +-----------------------------+                        |
|                      +---->+ ReinforcementLearningZoo.jl |                        |
|                            +----|------------------------+                        |
|                                 |                                                 |
|                                 |     +-------------------------------------+     |
|                                 +---->+ DistributedReinforcementLearning.jl |     |
|                                       +-------------------------------------+     |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### Scope of Each Package

- [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl)
  Two main concepts in reinforcement learning are precisely defined here: **Policy**
  and **Environment**.
- [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl)
  Typical environment examples in pure Julia and wrappers for 3-rd party
  environments are provided in this package.
- [ReinforcementLearningCore.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningCore.jl)
  Common utility functions and different layers of abstractions are contained in
  this package.
- [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl)
  Common reinforcement learning algorithms and their typical applications (aka
  `Experiment`s) are collected in this package.
- [DistributedReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/DistributedReinforcementLearning.jl)
  This package is still experimental and is not included in
  `ReinforcementLearning.jl` yet. Its goal is to extend some algorithms in
  `ReinforcementLearningZoo.jl` to apply them in distributed computing systems.

## Supporting 🖖

`ReinforcementLearning.jl` is a MIT licensed open source project with its
ongoing development made possible by many contributors in their spare time.
However, modern reinforcement learning research requires huge computing
resource, which is unaffordable for individual contributors. So if you or your
organization could provide the computing resource in some degree and would like
to cooperate in some way, please contact us!

## Citing

If you use `ReinforcementLearning.jl` in a scientific publication, we would
appreciate references to the following BibTex entry:

```
@misc{Tian2020Reinforcement,
  author       = {Jun Tian and other contributors},
  title        = {ReinforcementLearning.jl: A Reinforcement Learning Package for the Julia Language},
  year         = 2020,
  url          = {https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl}
}
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://lcn.epfl.ch/~brea/"><img src="https://avatars.githubusercontent.com/u/12857162?v=4?s=100" width="100px;" alt=""/><br /><sub><b>jbrea</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=jbrea" title="Code">💻</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=jbrea" title="Documentation">📖</a> <a href="#maintenance-jbrea" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://tianjun.me/"><img src="https://avatars.githubusercontent.com/u/5612003?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jun Tian</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=findmyway" title="Code">💻</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=findmyway" title="Documentation">📖</a> <a href="#maintenance-findmyway" title="Maintenance">🚧</a> <a href="#ideas-findmyway" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/amanbh"><img src="https://avatars.githubusercontent.com/u/911313?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Aman Bhatia</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=amanbh" title="Documentation">📖</a></td>
    <td align="center"><a href="https://avt.im/"><img src="https://avatars.githubusercontent.com/u/4722472?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alexander Terenin</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=aterenin" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Sid-Bhatia-0"><img src="https://avatars.githubusercontent.com/u/32610387?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sid-Bhatia-0</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=Sid-Bhatia-0" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/norci"><img src="https://avatars.githubusercontent.com/u/2986988?v=4?s=100" width="100px;" alt=""/><br /><sub><b>norci</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=norci" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/sriram13m"><img src="https://avatars.githubusercontent.com/u/28051516?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sriram</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=sriram13m" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/gpavanb1"><img src="https://avatars.githubusercontent.com/u/50511632?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Pavan B Govindaraju</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=gpavanb1" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/AlexLewandowski"><img src="https://avatars.githubusercontent.com/u/15149466?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alex Lewandowski</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=AlexLewandowski" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/RajGhugare19"><img src="https://avatars.githubusercontent.com/u/62653460?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Raj Ghugare</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=RajGhugare19" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/rbange"><img src="https://avatars.githubusercontent.com/u/13252574?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Roman Bange</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=rbange" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/felixchalumeau"><img src="https://avatars.githubusercontent.com/u/49362657?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Felix Chalumeau</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=felixchalumeau" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/rishabhvarshney14"><img src="https://avatars.githubusercontent.com/u/53183977?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Rishabh Varshney</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=rishabhvarshney14" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/zsunberg"><img src="https://avatars.githubusercontent.com/u/4240491?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zachary Sunberg</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=zsunberg" title="Code">💻</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=zsunberg" title="Documentation">📖</a> <a href="#maintenance-zsunberg" title="Maintenance">🚧</a> <a href="#ideas-zsunberg" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://www.cs.cmu.edu/~jlaurent/"><img src="https://avatars.githubusercontent.com/u/6361331?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jonathan Laurent</b></sub></a><br /><a href="#ideas-jonathan-laurent" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!