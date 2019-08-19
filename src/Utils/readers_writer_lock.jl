export ReaderCountRWLock, read_lock, read_unlock, is_read_locked

mutable struct ReaderCountRWLock <: Base.AbstractLock
    m::Threads.ReentrantLock
    reader_count::Int
    ReaderCountRWLock() = new(Threads.ReentrantLock(), 0)
end

function read_lock(l::ReaderCountRWLock)
    lock(l.m) do
        l.reader_count += 1
    end
end

function read_lock(f, l::ReaderCountRWLock)
    read_lock(l)
    try
        return f()
    finally
        read_unlock(l)
    end
end

function read_unlock(l::ReaderCountRWLock)
    lock(l.m) do
        l.reader_count -= 1
        if l.reader_count < 0
            error("reader count negative")
        end
    end
end

function Base.lock(l::ReaderCountRWLock)
    while true
        lock(l.m)
        if l.reader_count > 0
            unlock(l.m)
        else
            break
        end
    end
end

function Base.unlock(l::ReaderCountRWLock)
    unlock(l.m)
end

function Base.islocked(l::ReaderCountRWLock)
    islocked(l.m)
end

function is_read_locked(l::ReaderCountRWLock)
    l.reader_count > 0
end