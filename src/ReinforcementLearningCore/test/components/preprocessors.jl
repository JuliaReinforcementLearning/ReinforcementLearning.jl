@testset "preprocessors" begin

    @testset "ResizeImage" begin
        state = ones(4, 4)
        p = ResizeImage(2, 2)
        @test p(state) == ones(2, 2)
    end

    @testset "StackFrames" begin
        A = ones(2, 2)
        p = StackFrames(2, 2, 3)

        for i in 1:3
            p(A * i)
        end

        state = A * 4
        @test p(state) == reshape(repeat([2, 3, 4]; inner = 4), 2, 2, :)

    end

end
