# ReinforcementLearningBase.jl

[![Build Status](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl.svg?branch=master)](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl)

ReinforcementLearningBase.jl holds the common types and utility functions to be
shared by other components in ReinforcementLearning ecosystem.


## Examples

<table>
<th colspan="2">Traits</th><th> 1 </th><th> 2 </th><th> 3 </th><th> 4 </th><th> 5 </th><th> 6 </th><th> 7 </th><th> 8 </th><th> 9 </th><tr> <th rowspan="2"> ActionStyle </th><th> MinimalActionSet </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> FullActionSet </th><td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="3"> ChanceStyle </th><th> Stochastic </th><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> Deterministic </th><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> </tr>
<tr> <th> ExplicitStochastic </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="2"> DefaultStateStyle </th><th> Observation </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> </tr>
<tr> <th> InformationSet </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td></tr>
<tr> <th rowspan="2"> DynamicStyle </th><th> Simultaneous </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> Sequential </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="2"> InformationStyle </th><th> PerfectInformation </th><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> </tr>
<tr> <th> ImperfectInformation </th><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td></tr>
<tr> <th rowspan="2"> NumAgentStyle </th><th> MultiAgent </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> SingleAgent </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="2"> RewardStyle </th><th> TerminalReward </th><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> StepReward </th><td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="3"> StateStyle </th><th> Observation </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> </tr>
<tr> <th> InformationSet </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td></tr>
<tr> <th> InternalState </th><td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="4"> UtilityStyle </th><th> GeneralSum </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> ZeroSum </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> ✔ </td></tr>
<tr> <th> ConstantSum </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> </tr>
<tr> <th> IdenticalUtility </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> </tr>
</table>
<ol><li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/MultiArmBanditsEnv.jl"> MultiArmBanditsEnv </a></li>
<li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/RandomWalk1D.jl"> RandomWalk1D </a></li>
<li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/TigerProblemEnv.jl"> TigerProblemEnv </a></li>
<li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/MontyHallEnv.jl"> MontyHallEnv </a></li>
<li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/RockPaperScissorsEnv.jl"> RockPaperScissorsEnv </a></li>
<li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/TicTacToeEnv.jl"> TicTacToeEnv </a></li>
<li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/TinyHanabiEnv.jl"> TinyHanabiEnv </a></li>
<li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/PigEnv.jl"> PigEnv </a></li>
<li> <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/tree/master/src/examples/KuhnPokerEnv.jl"> KuhnPokerEnv </a></li>
</ol>