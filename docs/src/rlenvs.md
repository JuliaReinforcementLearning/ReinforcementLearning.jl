# ReinforcementLearningEnvironments.jl

## Built-in Environments

```@raw html
<table>
<th colspan="2">Traits</th><th> 1 </th><th> 2 </th><th> 3 </th><th> 4 </th><th> 5 </th><th> 6 </th><th> 7 </th><th> 8 </th><th> 9 </th><th> 10 </th><th> 11 </th><th> 12 </th><th> 13 </th><tr> <th rowspan="2"> ActionStyle </th><th> MinimalActionSet </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> FullActionSet </th><td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="3"> ChanceStyle </th><th> Stochastic </th><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> Deterministic </th><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> ExplicitStochastic </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="2"> DefaultStateStyle </th><th> Observation </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> InformationSet </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="2"> DynamicStyle </th><th> Simultaneous </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> Sequential </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="2"> InformationStyle </th><th> PerfectInformation </th><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> ImperfectInformation </th><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="2"> NumAgentStyle </th><th> MultiAgent </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> SingleAgent </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="2"> RewardStyle </th><th> TerminalReward </th><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> StepReward </th><td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="3"> StateStyle </th><th> Observation </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> InformationSet </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> InternalState </th><td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="4"> UtilityStyle </th><th> GeneralSum </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> ZeroSum </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> ConstantSum </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> IdenticalUtility </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
</table>
<ol><li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.MultiArmBanditsEnv-Tuple{}"> MultiArmBanditsEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.RandomWalk1D-Tuple{}"> RandomWalk1D </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.TigerProblemEnv-Tuple{}"> TigerProblemEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.MontyHallEnv-Tuple{}"> MontyHallEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.RockPaperScissorsEnv-Tuple{}"> RockPaperScissorsEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.TicTacToeEnv-Tuple{}"> TicTacToeEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.TinyHanabiEnv-Tuple{}"> TinyHanabiEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.PigEnv-Tuple{}"> PigEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.KuhnPokerEnv-Tuple{}"> KuhnPokerEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.AcrobotEnv-Tuple{}"> AcrobotEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.CartPoleEnv-Tuple{}"> CartPoleEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.MountainCarEnv-Tuple{}"> MountainCarEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.PendulumEnv-Tuple{}"> PendulumEnv </a></li>
</ol>
```

**Note**: Many traits are *borrowed* from [OpenSpiel](https://github.com/deepmind/open_spiel).

## 3-rd Party Environments

| Environment Name | Dependent Package Name | Description |
| :--- | :--- | :--- |
| `AtariEnv` | [ArcadeLearningEnvironment.jl](https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl) | |
| `GymEnv` | [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) | |
| `OpenSpielEnv` | [OpenSpiel.jl](https://github.com/JuliaReinforcementLearning/OpenSpiel.jl) | |
| `SnakeGameEnv` | [SnakeGames.jl](https://github.com/JuliaReinforcementLearning/SnakeGames.jl) | `SingleAgent`/`Multi-Agent`, `FullActionSet`/`MinimalActionSet`|
| [#list-of-environments](https://github.com/JuliaReinforcementLearning/GridWorlds.jl#list-of-environments) | [GridWorlds.jl](https://github.com/JuliaReinforcementLearning/GridWorlds.jl) | Environments in this package support the interfaces defined in `RLBase` |


```@autodocs
Modules = [ReinforcementLearningEnvironments]
```
