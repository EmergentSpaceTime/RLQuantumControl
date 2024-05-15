using Documenter
using DocumenterCitations

using RLQuantumControl


bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    ;
    sitename="RLQuantumControl.jl",
    modules=[RLQuantumControl],
    plugins=[bib],
    pages=[
        "Quick Start" => "index.md",
        "Agents" => [
            "Overview" => "agents/overview.md",
            "Algorithms" => "agents/algorithms.md",
        ],
        "Environments" => [
            "Overview" => "environments/overview.md",
            "Inputs" => "environments/inputs.md",
            "Shapings" => "environments/shapings.md",
            "Pulses" => "environments/pulses.md",
            "Models" => "environments/models.md",
            "Observations" => "environments/observations.md",
            "Rewards" => "environments/rewards.md",
            "Utility Functions" => "environments/utils.md",
        ],
    ],
    format=Documenter.HTML(
        ;
        mathengine=MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/physics"]),
                :tex => Dict(
                    "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                    "tags" => "ams",
                    "packages" => ["base", "ams", "autoload", "physics"],
                ),
            ),
        ),
        collapselevel=1,
        sidebar_sitename=true,
        ansicolor=true,
    )
)
