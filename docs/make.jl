push!(LOAD_PATH,"../src/")

#using FaSTLMMlight
using Documenter, FaSTLMMlight

#DocMeta.setdocmeta!(FaSTLMMlight, :DocTestSetup, :(using FaSTLMMlight); recursive=true)

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
        "Contents" => "index.md",
        "Introduction" => "introduction.md",
        "SVD of the fixed effects" => "svd-fixed-effects.md",
        "FaST-LMM full rank" => "fastlmm-fullrank.md",
        "List of functions" => "listfunctions.md"
    ],
)

deploydocs(;
    repo="github.com/tmichoel/FaSTLMMlight.jl.git",
    devbranch="main",
)
