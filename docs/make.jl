using Documenter, SailingRoutes

DocMeta.setdocmeta!(SailingRoutes, :DocTestSetup, :(using SailingRoutes); recursive=true)

makedocs(;
    modules=[SailingRoutes],
    authors="William Herrera",
    sitename="SailingRoutes.jl Documentation",
    repo=Documenter.Remotes.GitHub("wherrera10", "SailingRoutes.jl"),
    format=Documenter.HTML(;
        canonical="https://wherrera10.github.io/SailingRoutes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        # Add future pages here, e.g., "API Reference" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/wherrera10/SailingRoutes.jl.git",
    devbranch="main",
)

  
