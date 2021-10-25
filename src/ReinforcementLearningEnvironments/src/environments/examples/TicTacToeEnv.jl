export TicTacToeEnv

struct Nought end
const NOUGHT = Nought()
struct Cross end
const CROSS = Cross()

Base.:!(::Nought) = CROSS
Base.:!(::Cross) = NOUGHT

"""
This is a typical two player, zero sum game. Here we'll also demonstrate how to
implement an environment with multiple state representations.

You might be interested in this [blog](http://www.occasionalenthusiast.com/tag/tic-tac-toe/)
"""
mutable struct TicTacToeEnv <: AbstractEnv
    board::BitArray{3}
    player::Union{Nought,Cross}
end

function TicTacToeEnv()
    board = BitArray{3}(undef, 3, 3, 3)
    fill!(board, false)
    board[:, :, 1] .= true
    TicTacToeEnv(board, CROSS)
end

function RLBase.reset!(env::TicTacToeEnv)
    fill!(env.board, false)
    env.board[:, :, 1] .= true
    env.player = CROSS
end

struct TicTacToeInfo
    is_terminated::Bool
    winner::Union{Nothing,Nought,Cross}
end

const TIC_TAC_TOE_STATE_INFO = Dict{
    TicTacToeEnv,
    NamedTuple{
        (:index, :is_terminated, :winner),
        Tuple{Int,Bool,Union{Nothing,Nought,Cross}},
    },
}()

Base.hash(env::TicTacToeEnv, h::UInt) = hash(env.board, h)
Base.isequal(a::TicTacToeEnv, b::TicTacToeEnv) = isequal(a.board, b.board)

Base.to_index(::TicTacToeEnv, ::Cross) = 2
Base.to_index(::TicTacToeEnv, ::Nought) = 3

RLBase.action_space(::TicTacToeEnv, player) = Base.OneTo(9)

RLBase.legal_action_space(env::TicTacToeEnv, p) = findall(legal_action_space_mask(env))

function RLBase.legal_action_space_mask(env::TicTacToeEnv, p)
    if is_win(env, CROSS) || is_win(env, NOUGHT)
        zeros(false, 9)
    else
        vec(view(env.board, :, :, 1))
    end
end

(env::TicTacToeEnv)(action::Int) = env(CartesianIndices((3, 3))[action])

function (env::TicTacToeEnv)(action::CartesianIndex{2})
    env.board[action, 1] = false
    env.board[action, Base.to_index(env, env.player)] = true
    env.player = !env.player
end

RLBase.current_player(env::TicTacToeEnv) = env.player
RLBase.players(env::TicTacToeEnv) = (CROSS, NOUGHT)

RLBase.state(env::TicTacToeEnv, ::Observation{BitArray{3}}, p) = env.board
RLBase.state_space(env::TicTacToeEnv, ::Observation{BitArray{3}}, p) =
    Space(fill(false..true, 3, 3, 3))
RLBase.state(env::TicTacToeEnv, ::Observation{Int}, p) =
    get_tic_tac_toe_state_info()[env].index
RLBase.state_space(env::TicTacToeEnv, ::Observation{Int}, p) =
    Base.OneTo(length(get_tic_tac_toe_state_info()))

RLBase.state_space(env::TicTacToeEnv, ::Observation{String}, p) = WorldSpace{String}()

function RLBase.state(env::TicTacToeEnv, ::Observation{String}, p)
    buff = IOBuffer()
    for i in 1:3
        for j in 1:3
            if env.board[i, j, 1]
                x = '.'
            elseif env.board[i, j, 2]
                x = 'x'
            else
                x = 'o'
            end
            print(buff, x)
        end
        print(buff, '\n')
    end
    String(take!(buff))
end

RLBase.is_terminated(env::TicTacToeEnv) = get_tic_tac_toe_state_info()[env].is_terminated

function RLBase.reward(env::TicTacToeEnv, player)
    if is_terminated(env)
        winner = get_tic_tac_toe_state_info()[env].winner
        if isnothing(winner)
            0
        elseif winner === player
            1
        else
            -1
        end
    else
        0
    end
end

function is_win(env::TicTacToeEnv, player)
    b = env.board
    p = Base.to_index(env, player)
    @inbounds begin
        b[1, 1, p] & b[1, 2, p] & b[1, 3, p] ||
            b[2, 1, p] & b[2, 2, p] & b[2, 3, p] ||
            b[3, 1, p] & b[3, 2, p] & b[3, 3, p] ||
            b[1, 1, p] & b[2, 1, p] & b[3, 1, p] ||
            b[1, 2, p] & b[2, 2, p] & b[3, 2, p] ||
            b[1, 3, p] & b[2, 3, p] & b[3, 3, p] ||
            b[1, 1, p] & b[2, 2, p] & b[3, 3, p] ||
            b[1, 3, p] & b[2, 2, p] & b[3, 1, p]
    end
end

function get_tic_tac_toe_state_info()
    if isempty(TIC_TAC_TOE_STATE_INFO)
        @info "initializing state info..."
        t = @elapsed begin
            n = 1
            root = TicTacToeEnv()
            TIC_TAC_TOE_STATE_INFO[root] =
                (index = n, is_terminated = false, winner = nothing)
            walk(root) do env
                if !haskey(TIC_TAC_TOE_STATE_INFO, env)
                    n += 1
                    has_empty_pos = any(view(env.board, :, :, 1))
                    w = if is_win(env, CROSS)
                        CROSS
                    elseif is_win(env, NOUGHT)
                        NOUGHT
                    else
                        nothing
                    end
                    TIC_TAC_TOE_STATE_INFO[env] = (
                        index = n,
                        is_terminated = !(has_empty_pos && isnothing(w)),
                        winner = w,
                    )
                end
            end
        end
        @info "finished initializing state info in $t seconds"
    end
    TIC_TAC_TOE_STATE_INFO
end

RLBase.NumAgentStyle(::TicTacToeEnv) = MultiAgent(2)
RLBase.DynamicStyle(::TicTacToeEnv) = SEQUENTIAL
RLBase.ActionStyle(::TicTacToeEnv) = FULL_ACTION_SET
RLBase.InformationStyle(::TicTacToeEnv) = PERFECT_INFORMATION
RLBase.StateStyle(::TicTacToeEnv) =
    (Observation{String}(), Observation{Int}(), Observation{BitArray{3}}())
RLBase.RewardStyle(::TicTacToeEnv) = TERMINAL_REWARD
RLBase.UtilityStyle(::TicTacToeEnv) = ZERO_SUM
RLBase.ChanceStyle(::TicTacToeEnv) = DETERMINISTIC
