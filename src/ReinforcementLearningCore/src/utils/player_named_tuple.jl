import Base.getindex

struct PlayerNamedTuple{N,T}
    data::NamedTuple{N,T}

    function PlayerNamedTuple(data::Pair...)
        nt = NamedTuple(first(item).name => last(item) for item in data)
        new{typeof(nt).parameters...}(nt)
    end
end

Base.getindex(nt::PlayerNamedTuple, player::Player) = nt.data[player.name]
Base.keys(nt::PlayerNamedTuple) = Player.(keys(nt.data))
