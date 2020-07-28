"""
    run(π, env::AbstractEnv)

Run the policy `π` in `env` until the end.
"""
Base.run(π, env::AbstractEnv) = run(DynamicStyle(env), NumAgentStyle(env), π, env)

function Base.run(::Sequential, ::SingleAgent, π, env::AbstractEnv)
    while !get_terminal(env)
        action = π(env)
        env(action)
    end
end

function Base.run(::Sequential, ::MultiAgent, Π, env::AbstractEnv)
    is_terminal = false
    while !is_terminal
        for π in Π
            if get_terminal(env)
                is_terminal = true
                break
            end
            action = π(env)
            env(action)
        end
    end
end

#####
# printing
#####

function get_env_traits()
    [eval(x) for x in RLBase.ENV_API if endswith(String(x), "Style")]
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractEnv) =
    show(io, MIME"text/markdown"(), env)

function Base.show(io::IO, t::MIME"text/markdown", env::AbstractEnv)
    show(io, t, Markdown.parse("""
    # $(get_name(env))

    ## Traits
    | Trait Type | Value |
    |:---------- | ----- |
    $(join(["|$(string(f))|$(f(env))|" for f in get_env_traits()], "\n"))

    ## Actions
    $(get_actions(env))

    ## Players
    $(join(["- `$p`" for p in get_players(env)], "\n"))

    ## Current Player
    `$(get_current_player(env))`

    ## Is Environment Terminated?
    $(get_terminal(env) ? "Yes" : "No")
    """))
end
