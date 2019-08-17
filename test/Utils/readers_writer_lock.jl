@testset "readers_writer_lock" begin
    rwl = ReaderCountRWLock()

    @test islocked(rwl) == false
    @test is_read_locked(rwl) == false

    read_lock(rwl)
    @test is_read_locked(rwl) == true
    @test islocked(rwl) == false

    read_unlock(rwl)
    @test is_read_locked(rwl) == false
    @test islocked(rwl) == false

    lock(rwl)
    @test is_read_locked(rwl) == false
    @test islocked(rwl) == true

    unlock(rwl)
    @test is_read_locked(rwl) == false
    @test islocked(rwl) == false

    @test_throws ErrorException read_unlock(rwl)
end