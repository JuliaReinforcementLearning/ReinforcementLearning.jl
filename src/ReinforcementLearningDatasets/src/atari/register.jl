export atari_init
export atari_params

function atari_params()
    game = ATARI_GAMES
    index = 1:5
    epochs = 0:50
    @info game index epochs
end

const ATARI_GAMES = [
    "air-raid", "alien", "amidar", "assault", "asterix",
    "asteroids", "atlantis", "bank-heist", "battle-zone", "beam-rider",
    "berzerk", "bowling", "boxing", "breakout", "carnival", "centipede",
    "chopper-command", "crazy-climber", "demon-attack",
    "double-dunk", "elevator-action", "enduro", "fishing-derby", "freeway",
    "frostbite", "gopher", "gravitar", "hero", "ice-hockey", "jamesbond",
    "journey-escape", "kangaroo", "krull", "kung-fu-master",
    "montezuma-revenge", "ms-pacman", "name-this-game", "phoenix",
    "pitfall", "pong", "pooyan", "private-eye", "qbert", "riverraid",
    "road-runner", "robotank", "seaquest", "skiing", "solaris",
    "space-invaders", "star-gunner", "tennis", "time-pilot", "tutankham",
    "up-n-down", "venture", "video-pinball", "wizard-of-wor",
    "yars-revenge", "zaxxon"
]

game_name(game) = join(titlecase.(split(game, "-")))
# use function to initialise atari
function atari_init()
    for game in ATARI_GAMES
        for index in 1:5
            register(
                DataDep(
                    "atari-replay-datasets-$game-$index",
                    """
                    Dataset: The DQN Replay Dataset from Google Research
                    Authors: Rishabh Agarwal, Dale Schuurmans, Mohammad Norouzi
                    Publication Year: 2020
                    Related Publications: https://research.google/pubs/pub49020/
                    Credits: https://arxiv.org/abs/1907.04543
                    Journal: International Conference on Machine Learning

                    The DQN Replay Dataset is generated using DQN agents trained on 60 Atari 2600 
                    games for 200 million frames each, while using sticky actions (with 25% probability
                    that the agent’s previous action is executed instead of the current action) to make 
                    the problem more challenging. For each of the 60 games, we train 5 DQN agents with different
                    random initializations, and store all of the (state, action, reward, next state) tuples 
                    encountered during training into 5 replay datasets per game, resulting in a total of 300 datasets.
                    """,
                    "gs://atari-replay-datasets/dqn/$(game_name(game))/$index/replay_logs/";
                    fetch_method = fetch_gc_bucket
                )
            )
        end
    end
end