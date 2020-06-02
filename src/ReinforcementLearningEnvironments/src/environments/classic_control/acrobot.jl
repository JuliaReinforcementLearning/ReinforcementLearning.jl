using Random
using OrdinaryDiffEq

export AcrobotEnv


struct AcrobotEnvParams{T}
    link_length_a::T # [m]
    link_length_b::T # [m]
    link_mass_a::T #: [kg] mass of link 1
    link_mass_b::T #: [kg] mass of link 2
    #: [m] position of the center of mass of link 1
    link_com_pos_a::T
    #: [m] position of the center of mass of link 2
    link_com_pos_b::T
    #: Rotation related parameters
    link_moi::T
    max_torque_noise::T
    #: [m/s] maximum velocity of link 1
    max_vel_a::T
    #: [m/s] maximum velocity of link 2
    max_vel_b::T
    #: [m/s2] acceleration due to gravity
    g::T
    #: [s] timestep
    dt::T
    #: maximum steps in episode
    max_steps::Int
end

mutable struct AcrobotEnv{T,R<:AbstractRNG} <: AbstractEnv
    params::AcrobotEnvParams{T}
    action_space::DiscreteSpace{UnitRange{Int}}
    observation_space::MultiContinuousSpace{Vector{T}}
    state::Vector{T}
    action::Int
    done::Bool
    t::Int
    rng::R
    reward::T
    # difference in second link angular acceleration equation
    # as per python gym
    book_or_nips::String
    # array of available torques based on actions
    avail_torque::Vector{T}
end

"""
AcrobotEnv(;kwargs...)
# Keyword arguments
- `T = Float64`
- `link_length_a = T(1.)`
- `link_length_b = T(1.)`
- `link_mass_a = T(1.)`
- `link_mass_b = T(1.)`
- `link_com_pos_a = T(0.5)`
- `link_com_pos_b = T(0.5)`
- `link_moi = T(1.)`
- `max_vel_a = T(4 * π)`
- `max_vel_b = T(9 * π)`
- `g = T(9.8)`
- `dt = T(0.2)`
- `max_steps = 200`
- `book_or_nips = 'book'`
- `avail_torque = [T(-1.), T(0.), T(1.)]`
"""
function AcrobotEnv(;
    T = Float64,
    link_length_a = T(1.0),
    link_length_b = T(1.0),
    link_mass_a = T(1.0),
    link_mass_b = T(1.0),
    link_com_pos_a = T(0.5),
    link_com_pos_b = T(0.5),
    link_moi = T(1.0),
    max_torque_noise = T(0.0),
    max_vel_a = T(4 * π),
    max_vel_b = T(9 * π),
    g = T(9.8),
    dt = T(0.2),
    max_steps = 200,
    seed = nothing,
    book_or_nips = "book",
    avail_torque = [T(-1.0), T(0.0), T(1.0)],
)

    params = AcrobotEnvParams{T}(
        link_length_a,
        link_length_b,
        link_mass_a,
        link_mass_b,
        link_com_pos_a,
        link_com_pos_b,
        link_moi,
        max_torque_noise,
        max_vel_a,
        max_vel_b,
        g,
        dt,
        max_steps,
    )

    high = ([T(1.0), T(1.0), T(1.0), T(1.0), max_vel_a, max_vel_b])
    env = AcrobotEnv(
        params,
        DiscreteSpace(3),
        MultiContinuousSpace(-high, high),
        zeros(T, 4),
        0,
        false,
        0,
        MersenneTwister(seed),
        T(0.0),
        book_or_nips,
        [T(-1.0), T(0.0), T(1.0)],
    )
    reset!(env)
    env
end

acrobot_observation(s) = [cos(s[1]), sin(s[1]), cos(s[2]), sin(s[2]), s[3], s[4]]
angle_normalize(x) = ((x + pi) % (2 * pi)) - pi

RLBase.observe(env::AcrobotEnv) =
    (reward = env.reward, state = acrobot_observation(env.state), terminal = env.done)

function RLBase.reset!(env::AcrobotEnv{T}) where {T<:Number}
    env.state[:] = T(0.1) * rand(env.rng, T, 4) .- T(0.05)
    env.t = 0
    env.action = 2
    env.done = false
    nothing
end

# governing equations as per python gym
function (env::AcrobotEnv{T})(a) where {T<:Number}
    env.action = a
    env.t += 1
    torque = env.avail_torque[a]

    # noise to the force action
    noise_range = env.params.max_torque_noise
    if noise_range > 0
        torque += T(2.0 * noise_range) * rand(env.rng, T, 1) .- T(noise_range)
    end

    # augmented state for derivative function
    s_augmented = [env.state..., torque]

    ode = ODEProblem(dsdt, s_augmented, (0.0, env.params.dt), env)
    ns = solve(ode, RK4())
    # only care about final timestep of integration returned by integrator
    ns = ns.u[end]
    ns = ns[1:4]  # omit action

    # wrap the solution
    ns[1] = wrap(ns[1], -π, π)
    ns[2] = wrap(ns[2], -π, π)
    ns[3] = bound(ns[3], -env.params.max_vel_a, env.params.max_vel_a)
    ns[4] = bound(ns[4], -env.params.max_vel_b, env.params.max_vel_b)
    env.state = ns
    # termination criterion
    env.done = (-cos(ns[1]) - cos(ns[2] + ns[1]) > 1.0) || env.t > env.params.max_steps
    env.reward = env.done ? -1.0 : 0.0
    nothing
end

function dsdt(du, s_augmented, env::AcrobotEnv, t)
    # extract params
    m1 = env.params.link_mass_a
    m2 = env.params.link_mass_b
    l1 = env.params.link_length_a
    lc1 = env.params.link_com_pos_a
    lc2 = env.params.link_com_pos_b
    I1 = env.params.link_moi
    I2 = env.params.link_moi
    g = env.params.g

    # extract action and state
    a = s_augmented[end]
    s = s_augmented[1:end-1]

    # writing in standard form
    theta1 = s[1]
    theta2 = s[2]
    dtheta1 = s[3]
    dtheta2 = s[4]
    ddtheta1 = 0.0
    ddtheta2 = 0.0

    # governing equations
    d1 = (m1 * lc1^2 + m2 * (l1^2 + lc2^2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2)
    d2 = m2 * (lc2^2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2^2 * sin(theta2) -
        2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2) +
        (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) +
        phi2
    )
    if env.book_or_nips == "nips"
        # the following line is consistent with the description in the
        # paper
        ddtheta2 = ((a + d2 / d1 * phi1 - phi2) / (m2 * lc2^2 + I2 - d2^2 / d1))
    elseif env.book_or_nips == "book"
        # the following line is consistent with the java implementation and the
        # book
        ddtheta2 = (
            (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1^2 * sin(theta2) - phi2) /
            (m2 * lc2^2 + I2 - d2^2 / d1)
        )
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    end

    # return the values
    du[1] = dtheta1
    du[2] = dtheta2
    du[3] = ddtheta1
    du[4] = ddtheta2
    du[5] = 0.0
end

Random.seed!(env::AcrobotEnv, seed) = Random.seed!(env.rng, seed)

# wrap as per python gym
function wrap(x, m, M)
    """
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M
        x = x - diff
    end
    while x < m
        x = x + diff
    end
    return x
end

function bound(x, m, M)
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """

    # bound x between min (m) and Max (M)
    return min(max(x, m), M)
end
