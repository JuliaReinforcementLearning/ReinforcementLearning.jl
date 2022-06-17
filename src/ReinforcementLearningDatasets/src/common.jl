export SARTS
export SART
export SA
export RLDataSet
export fetch_gc_bucket
export fetch_gc_file

abstract type RLDataSet end

"""
`(:state, :action, :reward, :terminal, :next_state)`
type of the returned batches.
"""
const SARTS = (:state, :action, :reward, :terminal, :next_state)

"""
`(:state, :action, :reward, :terminal)`
type of the returned batches.
"""
const SART = (:state, :action, :reward, :terminal)

"""
`(:state, :action)`
type of the returned batches.
"""
const SA = (:state, :action)

# fetch_methods
"""
fetch a gc bucket from `src` to `dest`.
"""
function fetch_gc_bucket(src, dest)
    if Sys.iswindows()
        try run(`cmd /C gsutil -v`) catch x throw("gsutil not found, install gsutil to proceed further") end
        run(`cmd /C gsutil -m cp -r $src $dest`)
    else
        try run(`gsutil -v`) catch x throw("gsutil not found, install gsutil to proceed further") end
        run(`gsutil -m cp -r $src $dest`)
    end
    return dest
end

"""
fetch a gc file from `src` to `dest`.
"""
function fetch_gc_file(src, dest)
    if Sys.iswindows()
        try run(`cmd /C gsutil -v`) catch x throw("gsutil not found, install gsutil to proceed further") end
        run(`cmd /C gsutil -m cp $src $dest`)
    else
        try run(`gsutil -v`) catch x throw("gsutil not found, install gsutil to proceed further") end
        run(`gsutil -m cp $src $dest`)
    end
    return dest
end