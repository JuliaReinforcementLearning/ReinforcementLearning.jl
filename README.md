<!-- ```@raw html -->
<div align="center">
  <p>
  <img src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/master/docs/src/assets/logo.svg?sanitize=true" width="320px">
  </p>
  
  <p>
  <a href="https://wiki.c2.com/?MakeItWorkMakeItRightMakeItFast">"Make It Work Make It Right Make It Fast"</a>
  </p>
  
  <p>
  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/actions?query=workflow%3ACI"><img src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/workflows/CI/badge.svg"></a>
  <a href="https://juliahub.com/ui/Packages/ReinforcementLearning/6l2TO"><img src="https://juliahub.com/docs/ReinforcementLearning/pkgeval.svg"></a>
  <a href="https://juliahub.com/ui/Packages/ReinforcementLearning/6l2TO"><img src="https://juliahub.com/docs/ReinforcementLearning/version.svg"></a>
  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/LICENSE.md"><img src="http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat"></a>
  <a href="https://julialang.org/slack/"><img src="https://img.shields.io/badge/Chat%20on%20Slack-%23reinforcement--learnin-ff69b4"></a>
  <a href="https://github.com/SciML/ColPrac"><img src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet"></a>
  </p>

</div>
<!-- ``` -->

---

[**ReinforcementLearning.jl**](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl),
as the name says, is a package for reinforcement learning research in Julia.

Our design principles are:

- **Reusability and extensibility**: Provide elaborately designed components and
  interfaces to help users implement new algorithms.
- **Easy experimentation**: Make it easy for new users to run benchmark
  experiments, compare different algorithms, evaluate and diagnose agents.
- **Reproducibility**: Facilitate reproducibility from traditional tabular
  methods to modern deep reinforcement learning algorithms.
  

## 🏹 Get Started

```julia
julia> ] add ReinforcementLearning

julia> using ReinforcementLearning

julia> run(
           RandomPolicy(),
           CartPoleEnv(),
           StopAfterStep(1_000),
           TotalRewardPerEpisode()
       )
```

The above simple example demonstrates four core components in a general
reinforcement learning experiment:

- **Policy**. The
  [`RandomPolicy`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.RandomPolicy)
  is the simplest instance of
  [`AbstractPolicy`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.AbstractPolicy).
  It generates a random action at each step.

- **Environment**. The
  [`CartPoleEnv`](https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.CartPoleEnv-Tuple{})
  is a typical
  [`AbstractEnv`](https://juliareinforcementlearning.org/docs/rlbase/#ReinforcementLearningBase.AbstractEnv)
  to test reinforcement learning algorithms.

- **Stop Condition**. The
  [`StopAfterStep(1_000)`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.StopAfterStep)
  is to inform that our experiment should stop after
  `1_000` steps.

- **Hook**. The
  [`TotalRewardPerEpisode`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.TotalRewardPerEpisode)
  structure is one of the most common
  [`AbstractHook`](https://juliareinforcementlearning.org/docs/rlcore/#ReinforcementLearningCore.AbstractHook)s.
  It is used to collect the total reward of each episode in an experiment.

Check out the [tutorial](https://juliareinforcementlearning.org/docs/tutorial/) page to learn how these four
components are assembled together to solve many interesting problems. We also
write [blog](https://juliareinforcementlearning.org/blog/) occasionally to
explain the implementation details of some algorithms. Among them, the most
recommended one is [*An Introduction to
ReinforcementLearning.jl*](https://juliareinforcementlearning.org/blog/an_introduction_to_reinforcement_learning_jl_design_implementations_thoughts/),
which explains the design idea of this package. Besides, a collection of
[experiments](https://juliareinforcementlearning.org/docs/experiments/) are also provided to help you understand how to train
or evaluate policies, tune parameters, log intermediate data, load or save
parameters, plot results and record videos. For example:

<!-- ```@raw html -->
<img
src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/master/docs/src/assets/JuliaRL_BasicDQN_CartPole.gif?sanitize=true"
width="600px">

<!--

## 🙋 Why ReinforcementLearning.jl?

### 🚀 Fast Speed

[TODO:]

### 🧰 Feature Rich

[TODO:]

-->

<!-- ``` -->

## 🌲 Project Structure

`ReinforcementLearning.jl` itself is just a wrapper around several other
subpackages. The relationship between them is depicted below:

<!-- ```@raw html -->
<pre>+-----------------------------------------------------------------------------------+
|                                                                                   |
|  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl">ReinforcementLearning.jl</a>                                                         |
|                                                                                   |
|      +------------------------------+                                             |
|      | <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningBase">ReinforcementLearningBase.jl</a> |                                             |
|      +----|-------------------------+                                             |
|           |                                                                       |
|           |     +--------------------------------------+                          |
|           +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningEnvironments">ReinforcementLearningEnvironments.jl</a> |                          |
|           |     +--------------------------------------+                          |
|           |                                                                       |
|           |     +------------------------------+                                  |
|           +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningCore">ReinforcementLearningCore.jl</a> |                                  |
|                 +----|-------------------------+                                  |
|                      |                                                            |
|                      |     +-----------------------------+                        |
|                      +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningZoo">ReinforcementLearningZoo.jl</a> |                        |
|                            +----|------------------------+                        |
|                                 |                                                 |
|                                 |     +-------------------------------------+     |
|                                 +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/DistributedReinforcementLearning.jl">DistributedReinforcementLearning.jl</a> |     |
|                                       +-------------------------------------+     |
|                                                                                   |
+------|----------------------------------------------------------------------------+
       |
       |     +-------------------------------------+
       +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/tree/master/src/ReinforcementLearningExperiments">ReinforcementLearningExperiments.jl</a> |
       |     +-------------------------------------+
       |
       |     +----------------------------------------+
       +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl">ReinforcementLearningAnIntroduction.jl</a> |
             +----------------------------------------+

</pre>
<!-- ``` -->

## 🖖 Supporting

`ReinforcementLearning.jl` is a MIT licensed open source project with its
ongoing development made possible by many contributors in their spare time.
However, modern reinforcement learning research requires huge computing
resource, which is unaffordable for individual contributors. So if you or your
organization could provide the computing resource in some degree and would like
to cooperate in some way, please contact us!

This package is written in pure Julia. Please consider [supporting the JuliaLang org](https://github.com/sponsors/JuliaLang)
if you find this package useful. ❤

## ✍️ Citing

If you use `ReinforcementLearning.jl` in a scientific publication, we would
appreciate references to the [CITATION.bib](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/CITATION.bib).

## ✨ Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ```@raw html -->
<!-- cSpell:disable -->
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
    <td align="center"><a href="https://github.com/norci"><img src="https://avatars.githubusercontent.com/u/2986988?v=4?s=100" width="100px;" alt=""/><br /><sub><b>norci</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=norci" title="Code">💻</a> <a href="#maintenance-norci" title="Maintenance">🚧</a></td>
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
    <td align="center"><a href="https://github.com/drozzy"><img src="https://avatars.githubusercontent.com/u/140710?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andriy Drozdyuk</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=drozzy" title="Documentation">📖</a></td>
    <td align="center"><a href="http://ritchielee.net"><img src="https://avatars.githubusercontent.com/u/7119868?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ritchie Lee</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Arcnlee" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/xiruizhao"><img src="https://avatars.githubusercontent.com/u/35286069?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Xirui Zhao</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=xiruizhao" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/metab0t"><img src="https://avatars.githubusercontent.com/u/10501166?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nerd</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=metab0t" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/albheim"><img src="https://avatars.githubusercontent.com/u/3112674?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Albin Heimerson</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=albheim" title="Code">💻</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=albheim" title="Documentation">📖</a> <a href="#maintenance-albheim" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/michelangelo21"><img src="https://avatars.githubusercontent.com/u/49211663?v=4?s=100" width="100px;" alt=""/><br /><sub><b>michelangelo21</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Amichelangelo21" title="Bug reports">🐛</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/pilgrimygy"><img src="https://avatars.githubusercontent.com/u/49673553?v=4?s=100" width="100px;" alt=""/><br /><sub><b>GuoYu Yang</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=pilgrimygy" title="Documentation">📖</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=pilgrimygy" title="Code">💻</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Apilgrimygy" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/Mobius1D"><img src="https://avatars.githubusercontent.com/u/49596933?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Prasidh Srikumar</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=Mobius1D" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ilancoulon"><img src="https://avatars.githubusercontent.com/u/764934?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ilan Coulon</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=ilancoulon" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/JinraeKim"><img src="https://avatars.githubusercontent.com/u/43136096?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jinrae Kim</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=JinraeKim" title="Documentation">📖</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3AJinraeKim" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/luigiannelli"><img src="https://avatars.githubusercontent.com/u/24853508?v=4?s=100" width="100px;" alt=""/><br /><sub><b>luigiannelli</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Aluigiannelli" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/JBoerma"><img src="https://avatars.githubusercontent.com/u/7275916?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jacob Boerma</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=JBoerma" title="Code">💻</a></td>
    <td align="center"><a href="http://gitlab.com/plut0n"><img src="https://avatars.githubusercontent.com/u/50026682?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Xavier Valcarce</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Aplu70n" title="Bug reports">🐛</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://ashwani-rathee.github.io/"><img src="https://avatars.githubusercontent.com/u/54855463?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ashwani Rathee</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=ashwani-rathee" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jamblejoe"><img src="https://avatars.githubusercontent.com/u/12518354?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Goran Nakerst</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=jamblejoe" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ultradian"><img src="https://avatars.githubusercontent.com/u/14141325?v=4?s=100" width="100px;" alt=""/><br /><sub><b>ultradian</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=ultradian" title="Documentation">📖</a></td>
    <td align="center"><a href="https://bandism.net/"><img src="https://avatars.githubusercontent.com/u/22633385?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ikko Ashimine</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=eltociear" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/00krishna"><img src="https://avatars.githubusercontent.com/u/2063593?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Krishna Bhogaonker</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3A00krishna" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://www.is3.uni-koeln.de/de/team/doctoral-researchers/philipp-artur-kienscherf/"><img src="https://avatars.githubusercontent.com/u/44019953?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Philipp A. Kienscherf</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Apkienscherf" title="Bug reports">🐛</a></td>
    <td align="center"><a href="http://blog.krastanov.org/"><img src="https://avatars.githubusercontent.com/u/705248?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Stefan Krastanov</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=Krastanov" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/LaarsOman"><img src="https://avatars.githubusercontent.com/u/88617671?v=4?s=100" width="100px;" alt=""/><br /><sub><b>LaarsOman</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=LaarsOman" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/burmecia"><img src="https://avatars.githubusercontent.com/u/19306324?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Bo Lu</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=burmecia" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/peterchen96"><img src="https://avatars.githubusercontent.com/u/25033565?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Peter Chen</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=peterchen96" title="Code">💻</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=peterchen96" title="Documentation">📖</a></td>
    <td align="center"><a href="https://www.researchgate.net/profile/Shuhua_Gao2"><img src="https://avatars.githubusercontent.com/u/20141984?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Shuhua Gao</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=ShuhuaGao" title="Code">💻</a> <a href="#question-ShuhuaGao" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://github.com/johannes-fischer"><img src="https://avatars.githubusercontent.com/u/42044738?v=4?s=100" width="100px;" alt=""/><br /><sub><b>johannes-fischer</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=johannes-fischer" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/3rdCore"><img src="https://avatars.githubusercontent.com/u/59280588?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Tom Marty</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3A3rdCore" title="Bug reports">🐛</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=3rdCore" title="Code">💻</a></td>
    <td align="center"><a href="https://bhatiaabhinav.github.io/"><img src="https://avatars.githubusercontent.com/u/6555124?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Abhinav Bhatia</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Abhatiaabhinav" title="Bug reports">🐛</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=bhatiaabhinav" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://harwiltz.github.io/about"><img src="https://avatars.githubusercontent.com/u/56648659?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Harley Wiltzer</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=harwiltz" title="Code">💻</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=harwiltz" title="Documentation">📖</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Aharwiltz" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/dylan-asmar"><img src="https://avatars.githubusercontent.com/u/91484811?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Dylan Asmar</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=dylan-asmar" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/andreyzhitnikov"><img src="https://avatars.githubusercontent.com/u/20877529?v=4?s=100" width="100px;" alt=""/><br /><sub><b>andreyzhitnikov</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Aandreyzhitnikov" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/kir0ul"><img src="https://avatars.githubusercontent.com/u/6053592?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andrea PIERRÉ</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=kir0ul" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/Mo8it"><img src="https://avatars.githubusercontent.com/u/76752051?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mo8it</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=Mo8it" title="Code">💻</a></td>
    <td align="center"><a href="http://blegat.github.io"><img src="https://avatars.githubusercontent.com/u/1048205?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Benoît Legat</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=blegat" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/HenriDeh"><img src="https://avatars.githubusercontent.com/u/47037088?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Henri Dehaybe</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=HenriDeh" title="Code">💻</a> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=HenriDeh" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://nplawrence.com"><img src="https://avatars.githubusercontent.com/u/61165981?v=4?s=100" width="100px;" alt=""/><br /><sub><b>NPLawrence</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=NPLawrence" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/bileamScheuvens"><img src="https://avatars.githubusercontent.com/u/36153336?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Bileam Scheuvens</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=bileamScheuvens" title="Documentation">📖</a></td>
    <td align="center"><a href="http://jarbus.net"><img src="https://avatars.githubusercontent.com/u/42819002?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jarbus</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Ajarbus" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/tyleringebrand"><img src="https://avatars.githubusercontent.com/u/59975096?v=4?s=100" width="100px;" alt=""/><br /><sub><b>tyleringebrand</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues?q=author%3Atyleringebrand" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/baedan"><img src="https://avatars.githubusercontent.com/u/106585642?v=4?s=100" width="100px;" alt=""/><br /><sub><b>baedan</b></sub></a><br /><a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/commits?author=baedan" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- cSpell:enable -->
<!-- ``` -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
