using LinearAlgebra.BLAS: get_num_threads, set_num_threads

set_num_threads(Base.Threads.nthreads())
println(get_num_threads(), Base.Threads.nthreads())

using Random: seed!
using Flux: relu, glorot_uniform

using DelimitedFiles: readdlm
using TOML: parsefile
using HDF5: h5open, create_dataset, write, close
using BSON: @save, @load

include("../../src/RLQuantumControl.jl")
using .RLQuantumControl


function create_and_run_experiment(config::Dict{String, Vector})
    # Identity function.
    input_function = IdentityInput(
        3; control_min=fill(-5.4, 3), control_max=fill(2.4, 3)
    )
    # Shaping function.
    if config["shaping"] == "none"
        shaping_function = IdentityShaping(3, config["inputs"])
    elseif config["shaping"] == "fir"
        response_data = readdlm("response_data.txt")
        response_data[:, 2] .*= 0.020586246976990772 .* 1.0569028985914874
        shaping_function = FilterShaping(
            3,
            config["inputs"],
            Spline1D(response_data[:, 1], response_data[:, 2]; bc="zero");
            boundary_values=hcat(fill(-5.4, 3), fill(-5.4, 3)),
            sampling_rate=config["srate"],
        )
    elseif config["shaping"] == "gauss"
        mu = 1.9
        scale = 0.91
        sigma = 0.5
        shaping_function = FilterShaping(
            3,
            config["inputs"],
            Spline1D(
                -5:0.1:5,
                @. (
                    scale * exp(-0.5 * (((-5:0.1:5) - mu) / sigma) ^ 2)
                    / (sigma * sqrt(2π))
                );
                bc="zero",
            );
            boundary_values=hcat(fill(-5.4, 3), fill(-5.4, 3)),
            sampling_rate=config["srate"],
        )
    else
        throw(ArgumentError("Unsupported shaping function in config."))
    end
    # Model function.
    model_function = QuantumDot2(
        ;
        delta_t=(
            config["plength"]
            / (
                hasfield(typeof(shaping_function), :shaped_pulse_history)
                ? size(shaping_function.shaped_pulse_history, 2)
                : size(shaping_function.pulse_history, 2)
            )
        ),
        sigma_b=(
            config["reward"] == "robust" ? 0.0 : config["ndrift"] * 0.0105557513
        ),
    )
    # Pulse function.
    if config["pulse"] == "none"
        pulse_function = ExponentialPulse()
    elseif config["pulse"] == "both"
        if config["reward"] != "robust"
            pulse_function = Chain(
                StaticNoiseInjection(3, config["noises_epsilon_s"] * 0.0294),
                ColouredNoiseInjection(
                    3,
                    (
                        config["shaping"] == "none" ?
                        config["inputs"] :
                        (config["inputs"] + 5) * config["sampling_rate"]
                    ),
                    (
                        config["noises_epsilon_f"]
                        * 8e-16
                        * (1 / 0.272e-3 ^ 2)
                        * (1 / (model_function.delta_t * 1e-9)) ^ (1 - 0.7)
                    ),
                    0.7,
                ),
                ExponentialPulse(),
            )
        else
            pulse_function = ExponentialPulse()
        end
    else
        throw(ArgumentError("Unsupported pulse function in config."))
    end
    # Observation function.
    if config["observation"] == "full"
        observation_function = FullObservation()
    elseif config["observation"] == "noisy"
        observation_function = UnitaryTomography(3, "process", 6, true)
    end
    # Reward function.
    if config["reward"] == "sparse"
        reward_function = NormalisedReward(
            SparseGateFidelity([1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]), 0.99
        )
    elseif config["reward"] == "robust"
        reward_function = NormalisedReward(
            RobustGateFidelity(
                [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0],
                QuantumDot2(
                    ;
                    delta_t=model_function.delta_t,
                    sigma_b=config["noises_drift"] * 0.010555751316062399
                ),
                shaping_function.shaped_pulse_history,
                Chain(
                    StaticNoiseInjection(
                        3, config["noises_epsilon_s"] * 0.0294
                    ),
                    ColouredNoiseInjection(
                        3,
                        (
                            config["shaping"] == "none" ?
                            config["inputs"] :
                            (config["inputs"] + 5) * config["sampling_rate"]
                        ),
                        (
                            config["noises_epsilon_f"]
                            * 8e-16
                            * (1 / 0.272e-3 ^ 2)
                            * (1 / (model_function.delta_t * 1e-9)) ^ (1 - 0.7)
                        ),
                        0.7,
                    ),
                    ExponentialPulse(),
                ),
            ),
            0.99,
        )
    end
    env = QuantumControlEnvironment(
        ;
        input_function=input_function,
        shaping_function=shaping_function,
        pulse_function=pulse_function,
        model_function=model_function,
        observation_function=observation_function,
        reward_function=reward_function,
    )
    agent = SACAgent(
        env;
        activation=relu,
        init=glorot_uniform,
        capacity=100000,
        hiddens=[512, 512],
        log_var_min=-15,
        log_var_max=4,
        use_tqc=true,
        n_q=25,
        k_q=46,
        dropout=0.01,
        layer_norm=true,
        gamma=0.99,
        minibatch_size=256,
        training_steps=20,
        decays=[0.0, 0.0, 0.0, 0.0],
        clips=[5.0, 5.0, 5.0, 5.0],
        eta=[5e-4, 5e-4, 5e-4, 5e-4],
        rho=0.005,
        warmup_normalisation_episodes=150,
        warmup_evaluation_episodes=150,
        episodes=config["nmax"],  # Max episodes to run.
    )
    learn!(agent, env)
    return nothing
end


function continue_and_run_experiment(config::Dict{String, Vector})

end
# @save prefix * "_environment.bson" env
# @save prefix * "_agent.bson" agent

# d_file = h5open(prefix * "_data=0.h5", "cw")
# fset = create_dataset(d_file, "r", eltype(r), size(r))
# fset = create_dataset(d_file, "l", eltype(l), size(l))
# write(d_file["r"], r)
# write(d_file["l"], l)
# close(d_file)

# #################
# # Setup config. #
# #################
# CONFIG = parsefile(ARGS[1])

# # Setup seed and custom labels.
# seed!(CONFIG["seed"])

# ##################################
# # Initialisation of environment. #
# ##################################