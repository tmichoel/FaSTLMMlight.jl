push!(LOAD_PATH,"../src/")

#using SpatialOmicsGPs
using Documenter, FaSTLMMlight

#DocMeta.setdocmeta!(SpatialOmicsGPs, :DocTestSetup, :(using SpatialOmicsGPs); recursive=true)

makedocs(;
    modules=[FaSTLMMlight],
    sitename="FaST-LMM light",
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://tmichoel.github.io/FaSTLMMlight.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Introduction" => "index.md",
        "FaST-LMM full rank" => "FaST-LMM-fullrank.md",
        "List of functions" => "listfunctions.md"
    ],
)

deploydocs(;
    repo="github.com/tmichoel/FaSTLMMlight.jl.git",
    devbranch="main",
)
