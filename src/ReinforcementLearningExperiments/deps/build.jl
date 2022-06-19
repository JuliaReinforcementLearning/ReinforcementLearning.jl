using Weave

for (root, dirs, files) in walkdir(joinpath(@__DIR__, "experiments"))
    for f in files
        if splitext(f)[2] == ".jl"
            src = joinpath(root, f)
            dest_dir = joinpath(@__DIR__, "..", "src", "experiments")
            @info "extracting source code from $src to $dest_dir"
            tangle(src; informat="script", out_path=dest_dir)
        end
    end
end