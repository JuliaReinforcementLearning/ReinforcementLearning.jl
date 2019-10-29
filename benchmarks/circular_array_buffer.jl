using ReinforcementLearning, ReinforcementLearningEnvironments, Flux, BenchmarkTools

b = CircularArrayBuffer{Float64}(84, 84, 10)

function show_msg(desc, x)
    println(repeat("=", 50))
    println(desc)
    println(repeat("=", 50))
    display(x)
    println(repeat("=", 50))
    println()
end

show_msg("push a frame into buffer", @benchmark(push!($b, $(rand(84, 84)))))
show_msg("view a frame", @benchmark(select_frame($b, $(rand(1:10)))))
show_msg("consecutive view a batch", @benchmark(consecutive_view($b, $(rand(1:5, 32)), 4)))
show_msg("get an element", @benchmark(getindex($b, $(rand(1:84)), $(rand(1:84)), $(rand(1:10)))))

m = Dense(84*84, 512)

show_msg("view a frame and do matrix mul", @benchmark($m(vec(select_frame($b, $(rand(1:10)))))))
show_msg("view a frame and do matrix mul (use CartesianIndex)", @benchmark($m(vec(view($b, :, :, $(rand(1:10)))))))
show_msg("get a frame and do matrix mul", @benchmark($m(vec(getindex($b, :, :, $(rand(1:10)))))))
show_msg("do matrix mul", @benchmark($m(reshape($b, :, 10))))
show_msg("do matrix mul(for comparison)", @benchmark($m(reshape($(rand(84, 84, 10)), :, 10))))

#=
==================================================
push a frame into buffer
==================================================
BenchmarkTools.Trial: 
  memory estimate:  128 bytes
  allocs estimate:  4
  --------------
  minimum time:     1.449 μs (0.00% GC)
  median time:      1.472 μs (0.00% GC)
  mean time:        1.489 μs (0.00% GC)
  maximum time:     7.425 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     10
==================================================

==================================================
view a frame
==================================================
BenchmarkTools.Trial: 
  memory estimate:  64 bytes
  allocs estimate:  1
  --------------
  minimum time:     8.857 ns (0.00% GC)
  median time:      11.057 ns (0.00% GC)
  mean time:        19.323 ns (22.62% GC)
  maximum time:     4.618 μs (99.41% GC)
  --------------
  samples:          10000
  evals/sample:     999
==================================================

==================================================
consecutive view a batch
==================================================
BenchmarkTools.Trial: 
  memory estimate:  4.59 KiB
  allocs estimate:  80
  --------------
  minimum time:     1.331 μs (0.00% GC)
  median time:      1.411 μs (0.00% GC)
  mean time:        1.923 μs (20.34% GC)
  maximum time:     585.943 μs (99.59% GC)
  --------------
  samples:          10000
  evals/sample:     10
==================================================

==================================================
get an element
==================================================
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     3.617 ns (0.00% GC)
  median time:      3.637 ns (0.00% GC)
  mean time:        3.701 ns (0.00% GC)
  maximum time:     21.961 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1000
==================================================

==================================================
view a frame and do matrix mul
==================================================
BenchmarkTools.Trial: 
  memory estimate:  31.98 KiB
  allocs estimate:  6
  --------------
  minimum time:     294.195 μs (0.00% GC)
  median time:      318.465 μs (0.00% GC)
  mean time:        330.413 μs (0.57% GC)
  maximum time:     8.884 ms (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
==================================================

==================================================
view a frame and do matrix mul (use CartesianIndex)
==================================================
BenchmarkTools.Trial: 
  memory estimate:  33.33 KiB
  allocs estimate:  52
  --------------
  minimum time:     354.701 μs (0.00% GC)
  median time:      380.825 μs (0.00% GC)
  mean time:        393.057 μs (0.53% GC)
  maximum time:     8.943 ms (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
==================================================

==================================================
get a frame and do matrix mul
==================================================
BenchmarkTools.Trial: 
  memory estimate:  87.95 KiB
  allocs estimate:  39
  --------------
  minimum time:     342.392 μs (0.00% GC)
  median time:      367.611 μs (0.00% GC)
  mean time:        381.534 μs (1.16% GC)
  maximum time:     8.892 ms (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
==================================================

==================================================
do matrix mul
==================================================
BenchmarkTools.Trial: 
  memory estimate:  316.27 KiB
  allocs estimate:  17
  --------------
  minimum time:     1.307 ms (0.00% GC)
  median time:      1.354 ms (0.00% GC)
  mean time:        1.398 ms (0.90% GC)
  maximum time:     10.018 ms (0.00% GC)
  --------------
  samples:          3547
  evals/sample:     1
==================================================

==================================================
do matrix mul(for comparison)
==================================================
BenchmarkTools.Trial: 
  memory estimate:  315.95 KiB
  allocs estimate:  8
  --------------
  minimum time:     1.121 ms (0.00% GC)
  median time:      1.163 ms (0.00% GC)
  mean time:        1.202 ms (1.02% GC)
  maximum time:     10.235 ms (0.00% GC)
  --------------
  samples:          4121
  evals/sample:     1
==================================================
=#