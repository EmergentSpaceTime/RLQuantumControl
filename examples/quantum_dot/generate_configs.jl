using TOML


function generate_configs(path::String)
    parameters_variations = Dict{String, Vector}(
        "seed" => [1338, 24428, 322, 441],
        "inputs" => [15],
        "scale" => [0.91],
        "sigma" => [0.1],
        "shaping" => ["fir"],  # Kernel for shaping.
        "pulse" => ["none"],
        "nepss" => [1.0],  # "Fast" pulse noise.
        "nepsf" => [1.0],  # "Slow" pulse noise.
        "ndrift" => [1.0],  # Drift noise.
        "reward" => ["robust"],
        "observation" => ["full"],
        "nmeas" => [Int(1e10), Int(1e4)],  # Number of measurements.
        "plength" => [30],  # Protocol length.
        "srate" => [10],  # Sampling rate.
    )

    experiment_number = 0
    for param_values in Iterators.product(values(parameters_variations)...)
        params = Dict(zip(keys(parameters_variations), param_values))
        open(
            path * 
            "/config_"
            * string(experiment_number)
            * ".toml",
            "w",
        ) do io
            TOML.print(io, params)
        end
        experiment_number += 1
    end
    println("Generated $experiment_number configurations.")
    return nothing
end


generate_configs(ARGS[1])
