using .ArcadeLearningEnvironment
using .ReinforcementLearningEnvironments

using Dates
using ReinforcementLearningCore
using Flux
using BSON
using TensorBoardLogger
using Logging
using Statistics
using Random
using Setfield:@set

function atari_env_factory(
    name,
    state_size,
    n_frames,
    max_episode_steps = 100_000;
    seed = nothing,
    repeat_action_probability = 0.25,
)
    AtariEnv(;
        name = string(name),
        grayscale_obs = true,
        noop_max = 30,
        frame_skip = 4,
        terminal_on_life_loss = false,
        repeat_action_probability = repeat_action_probability,
        max_num_frames_per_episode = n_frames * max_episode_steps,
        color_averaging = false,
        full_action_space = false,
        seed = seed,
    ) |>
    StateOverriddenEnv(
        ResizeImage(state_size...),  # this implementation is different from cv2.resize https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py#L629
        StackFrames(state_size..., n_frames),
    ) |>
    StateCachedEnv |>
    RewardOverriddenEnv(r -> clamp(r, -1, 1))
end

for f in readdir(@__DIR__)
    if f != splitdir(@__FILE__)[2]
        include(f)
    end
end
