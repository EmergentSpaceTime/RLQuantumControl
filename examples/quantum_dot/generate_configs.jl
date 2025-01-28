using TOML: print


function create_folder(path::String)
    isdir(path) || mkpath(path)
    return nothing
end


function generate_configs(
    path::String,
    base_config::Dict{String, Vector},
    base_params::Vector{String} = String[],
)
    # Edit this to include more/less base parameters in namings.
    for b_p in base_params
        if !isone(length(base_config[b_p]))
            throw(
                ArgumentError(
                    "Base parameter $b_p must have only one value."
                )
            )
        end
    end

    unique_params = []
    for k_v in base_config
        length(k_v[2]) > 1 && append!(unique_params, [k_v[1]])
    end

    # If not using any unique parameters, add "seed" to base.
    if isempty(unique_params) & isempty(base_params)
        append!(base_params, ["seed"])
    end
    n_experiments = 0
    for v in Iterators.product(values(base_config)...)
        config = Dict(zip(keys(base_config), v))

        base_folder_name = join(
            [b_p * "=" * string(config[b_p]) for b_p in base_params], "_"
        )
        unique_folder_name = join(
            [u_p * "=" * string(config[u_p]) for u_p in unique_params], "_"
        )
        join_char = isempty(unique_params) | isempty(base_params) ? "" : "_"
        config_folder_name = base_folder_name * join_char * unique_folder_name
        config_folder_path = path * "/" * config_folder_name * "/"
        create_folder(config_folder_path)

        config["save_directory"] = config_folder_path
        open(config_folder_path * "config.toml", "w") do io
            print(io, config)
        end
        n_experiments += 1
    end
    println("Generated $n_experiments configurations.")
    return nothing
end


if !isone(length(ARGS))
    throw("Usage: `julia generate_configs.jl <data/dir_path>`")
end

base_config = Dict{String, Vector}(
    "seed" => [1338, 24428, 322, 441, 555, 666, 1996, 1453],
    "inputs" => [6, 16, 26, 36, 46],
    "sigma" => [0.5],
    "mu" => [1.35],
    "shaping" => ["fir"],  # Kernel for shaping.
    "pulse" => ["both"],
    "nepss" => [1.0],  # "Fast" pulse noise.
    "nepsf" => [1.0],  # "Slow" pulse noise.
    "ndrift" => [1.0],  # Drift noise.
    "reward" => ["robust"],
    "observation" => ["full"],
    "plength" => [10, 15, 20, 25, 30, 35, 40, 45],  # Protocol length.
    "srate" => [10],  # Oversampling rate.
    "nmeasures" => ["nothing"],  # Number of measurements.
    "save_directory" => ["results"],  # Dummy.
)
base_params = String[]
generate_configs(ARGS[1], base_config, base_params)
