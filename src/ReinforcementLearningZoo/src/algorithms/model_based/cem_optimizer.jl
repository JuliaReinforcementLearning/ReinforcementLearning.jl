export CEMTrajectoryOptimizer

mutable struct CEMTrajectoryOptimizer{T, R<:AbstractRNG} 
    iterations::Int
    population::Int
    elites::Int
    lower_bound::Vector{T}
    upper_bound::Vector{T}
    horizon::Int
    α::T
    σ_init::Vector{T}
    μ_last::Vector{T}
    rng::R
end

function CEMTrajectoryOptimizer(;
    lower_bound::Vector,
    upper_bound::Vector,
    iterations = 5,
    elite_ratio = 0.1,
    population = 350,
    α = 0.1,
    horizon = 15, 
    rng = Random.GLOBAL_RNG,
)
    CEMTrajectoryOptimizer(
        iterations, 
        population, 
        ceil(Int, population * elite_ratio),
        lower_bound,
        upper_bound,
        horizon,
        α,
        (upper_bound .- lower_bound) ./ 4,
        zeros(eltype(upper_bound), size(upper_bound)),
        rng,
    )
end

function (opt::CEMTrajectoryOptimizer)(fopt::F) where F
    μ = repeat(opt.μ_last, outer=(1, opt.horizon))
    σ = repeat(opt.σ_init, outer=(1, opt.horizon))

    best_solution = copy(μ)
    best_value = -Inf
    μ .= 0 # TODO
    for i in 1:opt.iterations
        margin_down = μ[:, 1] - opt.lower_bound
        margin_up = opt.upper_bound - μ[:, 1]
        σclipped = min(σ[:, 1], margin_down / 2, margin_up / 2)

        tvnorm = TruncatedNormal.(μ, σclipped, μ .- 2 .* σclipped, μ .+ 2 .* σclipped)
        population = [rand.(opt.rng, tvnorm) for _ in 1:opt.population]

        # make fopt take matrix
        values = fopt.(population)
        # Need NaN check and set to -Inf?

        elite_idxs = partialsortperm(values, 1:opt.elites; rev=true)
        elite = population[elite_idxs]
        best_values = values[elite_idxs]

        # Step μ and var towards new elites 
        μ = opt.α .* μ .+ (1 .- opt.α) .* mean(elite)
        σ = opt.α .* σ .+ (1 .- opt.α) .* std(elite, corrected=false)

        if best_values[1] > best_value
            best_value = best_values[1]
            best_solution = copy(elite[1])
        end
    end

    opt.μ_last = μ[:, 1]
    return opt.μ_last
end
