module ReinforcementLearningDatasets

const RLDatasets = ReinforcementLearningDatasets
export RLDatasets

using DataDeps

include("deep_ope/d4rl/d4rl_policies.jl")

include("d4rl/d4rl/register.jl")
include("d4rl/d4rl_pybullet/register.jl")
include("atari/register.jl")
include("rl_unplugged/atari/register.jl")
include("rl_unplugged/bsuite/register.jl")
include("rl_unplugged/dm/register.jl")
include("deep_ope/d4rl/register.jl")

include("common.jl")
include("init.jl")

include("d4rl/d4rl_dataset.jl")
include("atari/atari_dataset.jl")
include("rl_unplugged/util.jl")
include("rl_unplugged/atari/rl_unplugged_atari.jl")
include("rl_unplugged/bsuite/bsuite.jl")
include("rl_unplugged/dm/rl_unplugged_dm.jl")
include("deep_ope/d4rl/d4rl_policy.jl")

include("deep_ope/d4rl/evaluate.jl")

end