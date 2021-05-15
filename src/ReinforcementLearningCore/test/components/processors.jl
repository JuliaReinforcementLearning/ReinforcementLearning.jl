@testset "preprocessors" begin
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
