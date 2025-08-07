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
    ### Seed. ###
    "seed" => [24428, 555, 666, 1453],
    ### Environment. ###
    "inputs" => [16],
    # Filter related.
    "mu" => [1.013],  # Mean of Gaussian filter (default: 1.013).
    "sigma" => [0.6404],  # Standard deviation of filter (default: 0.6404).
    "skew" => [0.854],  # Skewness of shaping.
    "gaussamp" => [3.0607],  # Amplitude scaling of Gaussian (default: 3.0607).
    "filteramp" => [1.8],  # Amplitude scaling of filter.
    "shaping" => ["fir"],  # Kernel for shaping.
    "srate" => [10],  # Oversampling rate.
    # Pulse and noise related.
    "pulse" => ["both"],
    "directp" => [false],  # Direct pulse.
    "nepss" => [0.0],  # "Fast" pulse noise.
    "nepsf" => [0.0],  # "Slow" pulse noise.
    # Model related.
    "model" => ["qd1"],  # Model type.
    "plength" => [20],  # Protocol length.
    "tscale" => [1.0],  # Energy scaling.
    "ndrift" => [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],  # Drift noise.
    # Reward related.
    "reward" => ["robust"],  # Reward type.
    "loss" => ["inf"],
    "statfn" => ["logmean"],
    "normalreward" => [false],  # Normalise reward.
    "mapunitary" => [false],  # Map to unitary.
    "rmeasurement" => ["nothing"],
    "nruns" => [250],  # Number of runs for robust reward.
    "grscale" => [3.0029],
    # Observation related.
    "observation" => ["full"],  # Observation type.
    "normalobs" => [false],  # Normalise observation.
    "nmeasures" => ["nothing"],  # Number of measurements.
    ### Agent. ###
    "init" => ["gln"],  # Initialisation.
    "activation" => ["relu"],  # Activation.
    "tqc" => [true],  # TQC.
    "layernorm" => [true],  # Layer norm.
    "dropout" => [0.01],  # Dropout.
    "logvarmax" => [4],  # Log variance max.
    "lr" => [5e-4], # Learning rate.
    "initnormalepisodes" => [1],  # Evaluation steps.
    "initevalepisodes" => [150],  # Initial steps.
    "episodes" => [5000],  # Max number of episodes.
    # Agent saving.
    "savesteps" => [1000],  # How often to save data.
    "save_directory" => ["results"],  # Dummy.
)
base_params = String[]
generate_configs(ARGS[1], base_config, base_params)
