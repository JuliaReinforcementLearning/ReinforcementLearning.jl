function SnakeGameEnv(;action_style=MINIMAL_ACTION_SET,kw...)
    game = SnakeGame(;kw...)
    n_snakes = length(game.snakes)
    num_agent_style = n_snakes == 1 ? SINGLE_AGENT : MultiAgent{n_snakes}()
    SnakeGameEnv{action_style, num_agent_style, typeof(game)}(
        game,
        map(length, game.snakes),
        Vector{CartesianIndex{2}}(undef, length(game.snakes)),
        false
    )
end

RLBase.ActionStyle(env::SnakeGameEnv{A}) where A = A
RLBase.NumAgentStyle(env::SnakeGameEnv{<:Any, N}) where {N} = N
RLBase.DynamicStyle(env::SnakeGameEnv{<:Any, SINGLE_AGENT}) = SEQUENTIAL
RLBase.DynamicStyle(env::SnakeGameEnv{<:Any, <:MultiAgent}) = SIMULTANEOUS

const SNAKE_GAME_ACTIONS = (
    CartesianIndex(-1, 0),
    CartesianIndex(1, 0),
    CartesianIndex(0, 1),
    CartesianIndex(0, -1)
)

function (env::SnakeGameEnv{A})(actions::Vector{CartesianIndex{2}}) where {A}
    if A === MINIMAL_ACTION_SET
        # avoid turn back
        actions = [a_new == -a_old ? a_old : a_new for (a_new, a_old) in zip(actions, env.latest_actions)]
    end
    
    env.latest_actions .= actions
    map!(length, env.latest_snakes_length, env.game.snakes)
    env.is_terminated = !env.game(actions)
end

(env::SnakeGameEnv)(action::Int) = env([SNAKE_GAME_ACTIONS[action]])
(env::SnakeGameEnv)(actions::Vector{Int}) = env(map(a -> SNAKE_GAME_ACTIONS[a], actions))

RLBase.get_actions(env::SnakeGameEnv) = 1:4
RLBase.get_state(env::SnakeGameEnv) = env.game.board
RLBase.get_reward(env::SnakeGameEnv{<:Any, SINGLE_AGENT}) = length(env.game.snakes[]) - env.latest_snakes_length[]
RLBase.get_reward(env::SnakeGameEnv) = length.(env.game.snakes) .- env.latest_snakes_length
RLBase.get_terminal(env::SnakeGameEnv) = env.is_terminated

RLBase.get_legal_actions(env::SnakeGameEnv{FULL_ACTION_SET, SINGLE_AGENT}) = findall(!=(-env.latest_actions[]), SNAKE_GAME_ACTIONS)
RLBase.get_legal_actions(env::SnakeGameEnv{FULL_ACTION_SET}) = [findall(!=(-a), SNAKE_GAME_ACTIONS) for a in env.latest_actions]

RLBase.get_legal_actions_mask(env::SnakeGameEnv{FULL_ACTION_SET, SINGLE_AGENT}) = [a!=-env.latest_actions[] for a in SNAKE_GAME_ACTIONS]
RLBase.get_legal_actions_mask(env::SnakeGameEnv{FULL_ACTION_SET}) = [[x!=-a for x in SNAKE_GAME_ACTIONS] for a in env.latest_actions]

function RLBase.reset!(env::SnakeGameEnv)
    SnakeGames.reset!(env.game)
    env.is_terminated = false
    fill!(env.latest_actions, CartesianIndex(0,0))
    map!(length, env.latest_snakes_length, env.game.snakes)
end

Base.display(env::SnakeGameEnv) = display(env.game)
