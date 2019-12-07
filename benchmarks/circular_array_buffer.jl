using ReinforcementLearning, ReinforcementLearningEnvironments, Flux, BenchmarkTools, CuArrays

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
show_msg("view frames", @benchmark(select_frame($b, $(rand(1:10, 32)))))
show_msg("send frames to gpu", @benchmark(gpu(select_frame($b, $(rand(1:10, 32))))))
show_msg("consecutive view a batch", @benchmark(consecutive_view($b, $(rand(1:5, 32)), 4)))
show_msg("get an element", @benchmark(getindex($b, $(rand(1:84)), $(rand(1:84)), $(rand(1:10)))))

m = Dense(84*84, 512)

show_msg("view a frame and do matrix mul", @benchmark($m(vec(select_frame($b, $(rand(1:10)))))))
show_msg("view a frame and do matrix mul (use CartesianIndex)", @benchmark($m(vec(view($b, :, :, $(rand(1:10)))))))
show_msg("view frames and do matrix mul", @benchmark($m(reshape(select_frame($b, $(rand(1:10, 32))), :, 32))))
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
  minimum time:     1.429 μs (0.00% GC)
  median time:      1.449 μs (0.00% GC)
  mean time:        1.465 μs (0.00% GC)
  maximum time:     7.480 μs (0.00% GC)
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
  minimum time:     8.779 ns (0.00% GC)
  median time:      11.001 ns (0.00% GC)
  mean time:        19.588 ns (21.90% GC)
  maximum time:     4.445 μs (99.28% GC)
  --------------
  samples:          10000
  evals/sample:     999
==================================================

==================================================
view frames
==================================================
BenchmarkTools.Trial: 
  memory estimate:  496 bytes
  allocs estimate:  6
  --------------
  minimum time:     101.667 ns (0.00% GC)
  median time:      104.926 ns (0.00% GC)
  mean time:        138.549 ns (22.18% GC)
  maximum time:     4.617 μs (95.96% GC)
  --------------
  samples:          10000
  evals/sample:     940
==================================================

==================================================
send frames to gpu
==================================================
BenchmarkTools.Trial: 
  memory estimate:  553.31 KiB
  allocs estimate:  39
  --------------
  minimum time:     119.322 μs (0.00% GC)
  median time:      123.279 μs (0.00% GC)
  mean time:        140.553 μs (10.86% GC)
  maximum time:     2.255 ms (87.05% GC)
  --------------
  samples:          10000
  evals/sample:     1
==================================================

==================================================
consecutive view a batch
==================================================
BenchmarkTools.Trial: 
  memory estimate:  4.59 KiB
  allocs estimate:  80
  --------------
  minimum time:     1.377 μs (0.00% GC)
  median time:      1.474 μs (0.00% GC)
  mean time:        2.001 μs (19.49% GC)
  maximum time:     575.531 μs (99.30% GC)
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
  median time:      3.635 ns (0.00% GC)
  mean time:        3.657 ns (0.00% GC)
  maximum time:     19.967 ns (0.00% GC)
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
  minimum time:     316.350 μs (0.00% GC)
  median time:      338.291 μs (0.00% GC)
  mean time:        347.118 μs (0.54% GC)
  maximum time:     11.323 ms (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
==================================================

==================================================
view a frame and do matrix mul (use CartesianIndex)
==================================================
BenchmarkTools.Trial: 
  memory estimate:  33.30 KiB
  allocs estimate:  51
  --------------
  minimum time:     378.907 μs (0.00% GC)
  median time:      403.400 μs (0.00% GC)
  mean time:        413.936 μs (0.51% GC)
  maximum time:     11.349 ms (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
==================================================

==================================================
view frames and do matrix mul
==================================================
BenchmarkTools.Trial: 
  memory estimate:  1010.80 KiB
  allocs estimate:  13
  --------------
  minimum time:     3.211 ms (0.00% GC)
  median time:      3.261 ms (0.00% GC)
  mean time:        3.345 ms (1.07% GC)
  maximum time:     14.094 ms (0.00% GC)
  --------------
  samples:          1489
  evals/sample:     1
==================================================

==================================================
get a frame and do matrix mul
==================================================
BenchmarkTools.Trial: 
  memory estimate:  87.95 KiB
  allocs estimate:  39
  --------------
  minimum time:     379.192 μs (0.00% GC)
  median time:      401.962 μs (0.00% GC)
  mean time:        414.151 μs (1.11% GC)
  maximum time:     11.369 ms (0.00% GC)
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
  minimum time:     1.406 ms (0.00% GC)
  median time:      1.452 ms (0.00% GC)
  mean time:        1.492 ms (0.85% GC)
  maximum time:     12.403 ms (0.00% GC)
  --------------
  samples:          3323
  evals/sample:     1
==================================================

==================================================
do matrix mul(for comparison)
==================================================
BenchmarkTools.Trial: 
  memory estimate:  315.95 KiB
  allocs estimate:  8
  --------------
  minimum time:     1.198 ms (0.00% GC)
  median time:      1.262 ms (0.00% GC)
  mean time:        1.291 ms (0.97% GC)
  maximum time:     12.275 ms (0.00% GC)
  --------------
  samples:          3838
  evals/sample:     1
==================================================
=#