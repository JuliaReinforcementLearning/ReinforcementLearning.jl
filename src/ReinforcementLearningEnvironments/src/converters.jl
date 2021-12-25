export is_discrete_space, discrete2standard_discrete

is_discrete_space(x) = is_discrete_space(typeof(x))

is_discrete_space(::Type{<:AbstractVector}) = true
is_discrete_space(::Type{<:Tuple}) = true
is_discrete_space(::Type{<:NamedTuple}) = true

is_discrete_space(::Type) = false

"""
    discrete2standard_discrete(env)

Convert an `env` with a discrete action space to a standard form:

- The action space is of type `Base.OneTo`
- If the `env` is of `FULL_ACTION_SET`, then each action in the
  `legal_action_space(env)` is also an `Int` in the action space.

The standard form is useful for some algorithms (like Q-learning).
"""
function discrete2standard_discrete(env::AbstractEnv)
    A = action_space(env)
    if is_discrete_space(A)
        AS = ActionStyle(env)
        if AS === FULL_ACTION_SET
            mapping = Dict(x => i for (i, x) in enumerate(A))
            ActionTransformedEnv(
                env;
                action_space_mapping = a -> map(x -> mapping[x], a),
                action_mapping = i -> A[i],
            )
        elseif AS === MINIMAL_ACTION_SET
            ActionTransformedEnv(
                env;
                action_space_mapping = x -> Base.OneTo(length(A)),
                action_mapping = i -> A[i],
            )
        else
            @error "unknown ActionStyle $AS"
        end
    else
        throw(ArgumentError("unrecognized action space: $A"))
    end
end
