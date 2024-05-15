using TOML


parameters_variations = Dict{String, Vector}(
    "seed" => [1338, 24428, 1453, 5771],
    "inputs" => [45],
    "scale" => [0.91],
    "sigma" => [0.1],
    "shaping" => ["fir"],
    "pulse" => ["both"],
    "noises_epsilon" => [1, 0.1],
    "noises_drift" => [1, 0.5, 0.25, 0.1],
    "reward" => ["sparse", "robust"],
    "protocol_length" => [50],
    "sampling_rate" => [10],
)

experiment_number = 0
for param_values in Iterators.product(values(parameters_variations)...)
    params = Dict(zip(keys(parameters_variations), param_values))
    open(
        "examples/quantum_dot/configs/config_"
        * string(experiment_number)
        * ".toml",
        "w",
    ) do io
        TOML.print(io, params)
    end
    experiment_number += 1
end
