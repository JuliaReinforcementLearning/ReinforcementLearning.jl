@testset "preprocessors" begin

    @testset "ResizeImage" begin
        obs = (state = ones(4, 4),)
        p = ResizeImage(2, 2)
        @test get_state(p(obs)) == ones(2, 2)
    end

    @testset "StackFrames" begin
        A = ones(2, 2)
        p = StackFrames(2, 2, 3)

        for i in 1:3
            obs = (state = A * i,)
            p(obs)
        end

        obs = (state = A * 4,)
        @test get_state(p(obs)) == reshape(repeat([2, 3, 4]; inner = 4), 2, 2, :)

    end

end
