using ImageMattingApp
using Documenter

DocMeta.setdocmeta!(ImageMattingApp, :DocTestSetup, :(using ImageMattingApp); recursive=true)

makedocs(;
    modules=[ImageMattingApp],
    authors="Zheng-Kai Ye <supon3060@gmail.com> and contributors",
    repo="https://github.com/yezhengkai/ImageMattingApp.jl/blob/{commit}{path}#{line}",
    sitename="ImageMattingApp.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yezhengkai.github.io/ImageMattingApp.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yezhengkai/ImageMattingApp.jl",
    devbranch="main",
)
