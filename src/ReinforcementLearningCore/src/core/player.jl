struct Player <: AbstractPlayer
    name::Symbol

    function Player(name)
        new(Symbol(name))
    end
end
