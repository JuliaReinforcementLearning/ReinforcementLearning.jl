using Weave

const DEST_DIR = joinpath(@__DIR__, "..", "src", "experiments")

for (root, dirs, files) in walkdir(joinpath(@__DIR__, "experiments"))
    for f in files
        if splitext(f)[2] == ".jl"
            tangle(joinpath(root,f);informat="script", out_path=DEST_DIR)
        end
    end
end