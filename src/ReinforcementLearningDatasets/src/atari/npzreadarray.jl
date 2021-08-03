# Credits: https://github.com/fhs/NPZ.jl

# just a temporary file until NPZ.jl supports GZipStream

struct Header{T,N,F<:Function}
    descr::F
    fortran_order::Bool
    shape::NTuple{N,Int}
end

Header{T}(descr::F, fortran_order, shape::NTuple{N,Int}) where {T,N,F} = Header{T,N,F}(descr, fortran_order, shape)

const Numpy2Julia = Dict{String, DataType}()

const TypeMaps = [
    ("b1", Bool),
    ("i1", Int8),
    ("i2", Int16),
    ("i4", Int32),
    ("i8", Int64),
    ("u1", UInt8),
    ("u2", UInt16),
    ("u4", UInt32),
    ("u8", UInt64),
    ("f2", Float16),
    ("f4", Float32),
    ("f8", Float64),
    ("c8", Complex{Float32}),
    ("c16", Complex{Float64}),
]

function parseheader(s::AbstractString)
    s = parsechar(s, '{')

    dict = Dict{String,Any}()
    T = Any
    for _ in 1:3
        s = strip(s)
        key, s = parsestring(s)
        s = strip(s)
        s = parsechar(s, ':')
        s = strip(s)
        if key == "descr"
            (descr, T), s = parsedtype(s)
            dict[key] = descr
        elseif key == "fortran_order"
            dict[key], s = parsebool(s)
        elseif key == "shape"
            dict[key], s = parsetuple(s)
        else
            error("parsing header failed: bad dictionary key")
        end
        s = strip(s)
        if s[firstindex(s)] == '}'
            break
        end
        s = parsechar(s, ',')
    end
    s = strip(s)
    s = parsechar(s, '}')
    s = strip(s)
    if s != ""
        error("malformed header")
    end
    Header{T}(dict["descr"], dict["fortran_order"], dict["shape"])
end

function parsechar(s::AbstractString, c::Char)
    firstchar = s[firstindex(s)]
    if  firstchar != c
        error("parsing header failed: expected character '$c', found '$firstchar'")
    end
    SubString(s, nextind(s, 1))
end

function parsestring(s::AbstractString)
    s = parsechar(s, '\'')
    parts = split(s, '\'', limit = 2)
    length(parts) != 2 && error("parsing header failed: malformed string")
    parts[1], parts[2]
end

function parsebool(s::AbstractString)
    if SubString(s, firstindex(s), thisind(s, 4)) == "True"
        return true, SubString(s, nextind(s, 4))
    elseif SubString(s, firstindex(s), thisind(s, 5)) == "False"
        return false, SubString(s, nextind(s, 5))
    end
    error("parsing header failed: excepted True or False")
end

function parseinteger(s::AbstractString)
    isdigit(s[firstindex(s)]) || error("parsing header failed: no digits")
    tail_idx = findfirst(c -> !isdigit(c), s)
    if tail_idx == nothing
        intstr = SubString(s, firstindex(s))
    else
        intstr = SubString(s, firstindex(s), prevind(s, tail_idx))
        if s[tail_idx] == 'L' # output of firstindex should be a valid code point
            tail_idx = nextind(s, tail_idx)
        end
    end
    n = parse(Int, intstr)
    return n, SubString(s, tail_idx)
end

function parsetuple(s::AbstractString)
    s = parsechar(s, '(')
    tup = Int[]
    while true
        s = strip(s)
        if s[firstindex(s)] == ')'
            break
        end
        n, s = parseinteger(s)
        push!(tup, n)
        s = strip(s)
        if s[firstindex(s)] == ')'
            break
        end
        s = parsechar(s, ',')
    end
    s = parsechar(s, ')')
    Tuple(tup), s
end

function parsedtype(s::AbstractString)
    dtype, s = parsestring(s)
    c = dtype[firstindex(s)]
    t = SubString(dtype, nextind(s, 1))
    if c == '<'
        toh = ltoh
    elseif c == '>'
        toh = ntoh
    elseif c == '|'
        toh = identity
    else
        error("parsing header failed: unsupported endian character $c")
    end
    if !haskey(Numpy2Julia, t)
        error("parsing header failed: unsupported type $t")
    end
    (toh, Numpy2Julia[t]), s
end

function readheader(f::GZipStream)
    b = read!(f, Vector{UInt8}(undef, length(NPYMagic)))
    if b != NPYMagic
        error("not a numpy array file")
    end
    b = read!(f, Vector{UInt8}(undef, length(Version)))

    # support for version 2 files
    if b[1] == 1
        hdrlen = UInt32(readle(f, UInt16))
    elseif b[1] == 2 
        hdrlen = UInt32(readle(f, UInt32))
    else
        error("unsupported NPZ version")
    end

    hdr = ascii(String(read!(f, Vector{UInt8}(undef, hdrlen))))
    parseheader(strip(hdr))
end

function _npzreadarray(f, hdr::Header{T}) where {T}
    toh = hdr.descr
    if hdr.fortran_order
        x = map(toh, read!(f, Array{T}(undef, hdr.shape)))
    else
        x = map(toh, read!(f, Array{T}(undef, reverse(hdr.shape))))
        if ndims(x) > 1
            x = permutedims(x, collect(ndims(x):-1:1))
        end
    end
    ndims(x) == 0 ? x[1] : x
end

function npzreadarray(f::GZipStream)
    hdr = readheader(f)
    _npzreadarray(f, hdr)
end

for (s,t) in TypeMaps
    Numpy2Julia[s] = t
end

const Julia2Numpy = Dict{DataType, String}()