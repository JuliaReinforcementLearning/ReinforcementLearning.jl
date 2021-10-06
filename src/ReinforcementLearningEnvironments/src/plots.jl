import .Plots.plot
import .Plots.plot!

function plot(env::CartPoleEnv; kwargs...)
    s, a, d = env.state, env.action, env.done
    x, xdot, theta, thetadot = s
    l = 2 * env.params.halflength
    xthreshold = env.params.xthreshold
    # set the frame
    plot(
        xlims = (-xthreshold, xthreshold),
        ylims = (-.1, l + 0.1),
        legend = false,
        border = :none,
    )
    # plot the cart
    plot!([x - 0.5, x - 0.5, x + 0.5, x + 0.5], [-.05, 0, 0, -.05]; seriestype = :shape)
    # plot the pole
    plot!([x, x + l * sin(theta)], [0, l * cos(theta)]; linewidth = 3)
    # plot the arrow
    plot!(
        [x + (a == 1) - 0.5, x + 1.4 * (a == 1) - 0.7],
        [-.025, -.025];
        linewidth = 3,
        arrow = true,
        color = 2,
    )
    # if done plot pink circle in top right
    if d
        plot!(
            [xthreshold - 0.2],
            [l];
            marker = :circle,
            markersize = 20,
            markerstrokewidth = 0.0,
            color = :pink,
        )
    end

    plot!(; kwargs...)
end


# adapted from https://github.com/JuliaML/Reinforce.jl/blob/master/src/envs/mountain_car.jl
height(xs) = sin(3 * xs) * 0.45 + 0.55
rotate(xs, ys, θ) = xs * cos(θ) - ys * sin(θ), ys * cos(θ) + xs * sin(θ)
translate(xs, ys, t) = xs .+ t[1], ys .+ t[2]

function plot(env::MountainCarEnv; kwargs...)
    s = env.state
    d = env.done

    plot(
        xlims = (env.params.min_pos - 0.1, env.params.max_pos + 0.2),
        ylims = (-.1, height(env.params.max_pos) + 0.2),
        legend = false,
        border = :none,
    )
    # plot the terrain
    xs = LinRange(env.params.min_pos, env.params.max_pos, 100)
    ys = height.(xs)
    plot!(xs, ys)

    # plot the car
    x = s[1]
    θ = cos(3 * x)
    carwidth = 0.05
    carheight = carwidth / 2
    clearance = 0.2 * carheight
    xs = [-carwidth / 2, -carwidth / 2, carwidth / 2, carwidth / 2]
    ys = [0, carheight, carheight, 0]
    ys .+= clearance
    xs, ys = rotate(xs, ys, θ)
    xs, ys = translate(xs, ys, [x, height(x)])
    plot!(xs, ys; seriestype = :shape)

    # if done plot pink circle in top right
    if d
        plot!(
            [xthreshold - 0.2],
            [l];
            marker = :circle,
            markersize = 20,
            markerstrokewidth = 0.0,
            color = :pink,
        )
    end

    plot!(; kwargs...)
end
