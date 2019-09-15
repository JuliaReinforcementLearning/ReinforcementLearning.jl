export ParameterServer, update!

using Flux: Params

struct ParameterServer
    rwl::ReaderCountRWLock
    params::Params
    ParameterServer(ps) = new(ReaderCountRWLock(), ps)
end

function update!(f, ps::ParameterServer)
    lock(ps.rwl) do
        f(ps.params)
    end
end

function Base.get(f, ps::ParameterServer)
    read_lock(ps.rwl) do
        f(ps.params)
    end
end
