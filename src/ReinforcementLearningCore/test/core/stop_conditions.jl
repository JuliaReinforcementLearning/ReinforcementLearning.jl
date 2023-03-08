@testset "StopAfterStep" begin
    stop_ = StopAfterStep(10)
    @test sum([stop_() for i in 1:20]) == 11

    stop_ = StopAfterStep(10; is_show_progress=false)
    @test sum([stop_() for i in 1:20]) == 11
end
