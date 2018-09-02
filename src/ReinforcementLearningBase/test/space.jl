@testset "Space" begin

@testset "BoxSpace" begin
    @test occursin(0.5, BoxSpace(0, 1)) == true
    @test occursin(0.0, BoxSpace(0, 1)) == true
    @test occursin(1.0, BoxSpace(0, 1)) == true
    @test occursin(-1.0, BoxSpace(0, 1)) == false
    @test occursin(-Inf, BoxSpace(0, 1)) == false
    @test occursin([0.5], BoxSpace(0, 1)) == true

    @test occursin([0, 0], BoxSpace([-1, -2], [1, 2])) == true
    @test occursin([0, 3], BoxSpace([-1, -2], [1, 2])) == false
    @test occursin([0, 0], BoxSpace([-1, -2], [1, 2])) == true
end

@testset "DiscreteSpace" begin
    @test occursin(0, DiscreteSpace(10, 0)) == true
    @test occursin(5, DiscreteSpace(10, 0)) == true
    @test occursin(10, DiscreteSpace(10, 0)) == false
end

@testset "MultiBinarySpace" begin
    @test occursin([true false; true false], MultiBinarySpace(2,2)) == true
    @test occursin([true false], MultiBinarySpace(2,2)) == false
end

@testset "MultiDiscreteSpace" begin
    @test occursin(0, MultiDiscreteSpace([2,3,2], 0)) == true
    @test occursin(1, MultiDiscreteSpace([2,3,2], 0)) == true
    @test occursin(2, MultiDiscreteSpace([2,3,2], 0)) == false
    @test occursin([1,1,1], MultiDiscreteSpace([2,3,2], 0)) == true
    @test occursin([0,0,0], MultiDiscreteSpace([2,3,2], 0)) == true
    @test occursin([3,3,3], MultiDiscreteSpace([2,3,2], 0)) == false
end

@testset "Space Tuple" begin
    @test occursin(([0.5], 5, [true true; true true], [1, 1]),
                   (BoxSpace(0,1), DiscreteSpace(5, 0), MultiBinarySpace(2,2), MultiDiscreteSpace([2,2], 0))) == false
    @test occursin(([0.5], 0, [true true; true true], [1, 1]),
                   (BoxSpace(0,1), DiscreteSpace(5, 0), MultiBinarySpace(2,2), MultiDiscreteSpace([2,2], 0))) == true
    @test occursin((), 
                   (BoxSpace(0,1), DiscreteSpace(5, 0), MultiBinarySpace(2,2), MultiDiscreteSpace([2,2], 0))) == false
end

@testset "Space Dict" begin
    @test occursin(
        Dict(
            "sensors" => Dict(
                "position" => [-10, 0, 10],
                "velocity" => [0.1, 0.2, 0.3],
                "front_cam" => (rand(10, 10, 3), rand(10, 10, 3)),
                "rear_cam" => rand(10,10,3)),
            "ext_controller" => [2, 1, 1],
            "inner_state" => Dict(
                "charge" => 35,
                "system_checks" => rand(Bool, 10),
                "job_status" => Dict(
                    "task" => 3,
                    "progress" => 23))),
        Dict(
            "sensors" =>  Dict(
                "position"=> BoxSpace(-100, 100, (3,)),
                "velocity"=> BoxSpace(-1, 1, (3,)),
                "front_cam"=> (BoxSpace(0, 1, (10, 10, 3)),
                               BoxSpace(0, 1, (10, 10, 3))),
                "rear_cam" => BoxSpace(0, 1, (10, 10, 3))),
            "ext_controller" => MultiDiscreteSpace([5, 2, 2], 0),
            "inner_state" => Dict(
                "charge" => DiscreteSpace(100, 0),
                "system_checks" => MultiBinarySpace(10),
                "job_status" => Dict(
                    "task" => DiscreteSpace(5, 0),
                    "progress" => BoxSpace(0, 100))))) == true
end

end