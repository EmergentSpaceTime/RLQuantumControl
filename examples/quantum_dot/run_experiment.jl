using LinearAlgebra.BLAS: get_num_threads, set_num_threads

set_num_threads(Base.Threads.nthreads())
println(get_num_threads(), Base.Threads.nthreads())

using Dierckx: Spline1D
using DelimitedFiles: readdlm
using Distributions: pdf, SkewNormal
using Flux: relu, gelu, glorot_uniform, glorot_normal
using Random: seed!
using TOML: parsefile

using RLQuantumControl

#################
# Setup config. #
#################
CONFIG = parsefile(ARGS[1])
# CONFIG = parsefile(
#     "examples/quantum_dot/testing/"
#     * "plength=25_seed=45_inputs=26"
#     * "/config.toml"
# )
SEED = CONFIG["seed"]

# Setup seed and custom labels.
if "rewards.txt" in readdir(CONFIG["save_directory"])
# if "rewards.txt" in readdir("examples/quantum_dot/" * CONFIG["save_directory"])
    seed!(SEED + length(readdlm(CONFIG["save_directory"] * "rewards.txt")))
    # seed!(
    #     SEED
    #     + length(
    #         readdlm(
    #             "examples/quantum_dot/"
    #             * CONFIG["save_directory"]
    #             * "rewards.txt"
    #         )
    #     )
    # )
else
    seed!(SEED)
end

######################
# Setup environment. #
######################
if CONFIG["model"] == "qd1"
    N_CTRLS = 1
    TARGET = [1 0; 0 im]
elseif CONFIG["model"] == "qd2"
    N_CTRLS = 3
    TARGET = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
end
# Input function.
input_function = IdentityInput(
    N_CTRLS;
    control_min=fill(CONFIG["directp"] ? exp(-5.4) : -5.4, N_CTRLS),
    control_max=fill(CONFIG["directp"] ? exp(2.4) : 2.4, N_CTRLS),
)
# Shaping function.
if CONFIG["shaping"] == "none"
    shaping_function = IdentityShaping(N_CTRLS, CONFIG["inputs"])
elseif CONFIG["shaping"] == "fir"
    response_data = readdlm("response_data.txt")
    # response_data = readdlm("examples/quantum_dot/response_data.txt")
    response_data[:, 2] ./= (
        maximum(abs.(response_data[:, 2]))
        * CONFIG["srate"]
        * CONFIG["filteramp"]
    )
    shaping_function = FilterShaping(
        N_CTRLS,
        CONFIG["inputs"],
        Spline1D(response_data[:, 1], response_data[:, 2]; bc="zero");
        oversampling_rate=CONFIG["srate"],
        boundary_values=hcat(
            fill(CONFIG["directp"] ? exp(-5.4) : -5.4, N_CTRLS),
            fill(CONFIG["directp"] ? exp(-5.4) : -5.4, N_CTRLS),
        ),
        boundary_padding=[5, 4],
    )
elseif CONFIG["shaping"] == "gauss"
    TDKM = 0:0.01:10
    YDKM = pdf(SkewNormal(CONFIG["mu"], CONFIG["sigma"], CONFIG["skew"]), TDKM)
    YDKM ./= maximum(abs.(YDKM)) * CONFIG["srate"] * CONFIG["gaussamp"]
    shaping_function = FilterShaping(
        N_CTRLS,
        CONFIG["inputs"],
        Spline1D(TDKM, YDKM; bc="zero");
        oversampling_rate=CONFIG["srate"],
        boundary_values=hcat(
            fill(CONFIG["directp"] ? exp(-5.4) : -5.4, N_CTRLS),
            fill(CONFIG["directp"] ? exp(-5.4) : -5.4, N_CTRLS),
        ),
        boundary_padding=[5, 4],
    )
end
# Model function.
if CONFIG["model"] == "qd1"
    model_function = Simple1DSystem(
        (
            CONFIG["tscale"]
            * CONFIG["plength"]
            / (
                hasfield(typeof(shaping_function), :shaped_pulse_history)
                ? size(shaping_function.shaped_pulse_history, 2)
                : size(shaping_function.pulse_history, 2)
            )
        );
        j_0=1.0 / CONFIG["tscale"],
        b=2.09 / CONFIG["tscale"],
        sigma_b=(
            CONFIG["reward"] == "robust" ? 0.0 : CONFIG["ndrift"] * 0.0105
        ),
    )
elseif CONFIG["model"] == "qd2"
    model_function = QuantumDot2(
        (
            CONFIG["tscale"]
            * CONFIG["plength"]
            / (
                hasfield(typeof(shaping_function), :shaped_pulse_history)
                ? size(shaping_function.shaped_pulse_history, 2)
                : size(shaping_function.pulse_history, 2)
            )
        );
        b_ij=[1.0, 7.0, -1.0] ./ CONFIG["tscale"],
        j_0=1.0 / CONFIG["tscale"],
        sigma_b=(
            CONFIG["reward"] == "robust" ? 0.0 : CONFIG["ndrift"] * 0.0105
        ),
    )
end
# Pulse function.
if (CONFIG["pulse"] == "none") | (CONFIG["reward"] in ["robust", "robustg"])
    pulse_function = CONFIG["directp"] ? IdentityPulse() : ExponentialPulse()
elseif (CONFIG["pulse"] == "both") & !CONFIG["directp"]
    pulse_function = Chain(
        (
            iszero(CONFIG["nepss"])
            ? []
            : [StaticNoiseInjection(N_CTRLS, CONFIG["nepss"] * 0.0294)]
        )...,
        (
            iszero(CONFIG["nepsf"])
            ? []
            : [
                ColouredNoiseInjection(
                    N_CTRLS,
                    (
                        CONFIG["shaping"] == "none"
                        ? CONFIG["inputs"]
                        : (
                            (
                                CONFIG["inputs"]
                                + shaping_function.boundary_padding[2]
                            )
                            * CONFIG["srate"]
                        )
                    ),
                    0.7,
                    1 / (2π * CONFIG["tscale"] * model_function.delta_t * 1e-9),
                    CONFIG["nepsf"] * 8.57e-9,
                )
            ]
        )...,
        ExponentialPulse(),
    )
end
# Observation function.
if CONFIG["observation"] == "full"
    observation_function = FullObservation()
elseif CONFIG["observation"] == "pulse"
    observation_function = PulseHistory(N_CTRLS, CONFIG["inputs"])
elseif CONFIG["observation"] == "noisy"
    if CONFIG["nmeasures"] != "nothing"
        observation_function = UnitaryTomography(
            N_CTRLS,
            "unitary",
            (
                CONFIG["model"] == "qd1"
                ? 2
                : (
                    CONFIG["model"] == "qd2"
                    ? 6
                    : throw(ErrorException("Invalid model."))
                )
            ),
            true;
            n=CONFIG["nmeasures"],
        )
    else
        observation_function = FullObservation()
    end
elseif CONFIG["observation"] == "process"
    observation_function = ExactTomography(
        N_CTRLS,
        "process",
        (
            CONFIG["model"] == "qd1"
            ? 2
            : (
                CONFIG["model"] == "qd2"
                ? 6
                : throw(ErrorException("Invalid model."))
            )
        ),
        false,
    )
end
if CONFIG["normalobs"]
    observation_function = NormalisedObservation(
        observation_function,
        (
            CONFIG["model"] == "qd1"
            ? (isa(observation_function, FullObservation) ? 10 : 9)
            : (
                CONFIG["model"] == "qd2"
                ? (isa(observation_function, FullObservation) ? 76 : 73)
                : throw(ErrorException("Invalid model."))
            )
        ),
    )
end
# Reward function.
if CONFIG["reward"] == "sparse"
    reward_function = SparseGateFidelity(
        TARGET,
        (
            CONFIG["model"] == "qd1"
            ? nothing
            : (
                CONFIG["model"] == "qd2"
                ? UnitRange(2, 5)
                : throw(ErrorException("Invalid model."))
            )
        ),
        CONFIG["mapunitary"],
    )
elseif CONFIG["reward"] == "robust"
    reward_function = RobustGateReward(
        TARGET,
        (
            CONFIG["model"] == "qd1"
            ? Simple1DSystem(
                model_function.delta_t;
                j_0=1.0 / CONFIG["tscale"],
                b=2.09 / CONFIG["tscale"],
                sigma_b=CONFIG["ndrift"] * 0.0105,
            )
            : (
                CONFIG["model"] == "qd2"
                ? QuantumDot2(
                    model_function.delta_t;
                    j_0=1.0 / CONFIG["tscale"],
                    b_ij=[1.0, 7.0, -1.0] ./ CONFIG["tscale"],
                    sigma_b=CONFIG["ndrift"] * 0.0105,
                )
                : throw(ErrorException("Invalid model."))
            )
        ),
        shaping_function.shaped_pulse_history,
        (
            CONFIG["directp"]
            ? IdentityPulse()
            : (
                CONFIG["pulse"] == "none"
                ? ExponentialPulse()
                : Chain(
                    (
                        iszero(CONFIG["nepss"])
                        ? []
                        : [
                            StaticNoiseInjection(
                                N_CTRLS, CONFIG["nepss"] * 0.0294
                            )
                        ]
                    )...,
                    (
                        iszero(CONFIG["nepsf"])
                        ? []
                        : [
                            ColouredNoiseInjection(
                                N_CTRLS,
                                (
                                    CONFIG["shaping"] == "none"
                                    ? CONFIG["inputs"]
                                    : (
                                        (
                                            CONFIG["inputs"]
                                            + shaping_function.boundary_padding[
                                                2
                                            ]
                                        )
                                        * CONFIG["srate"]
                                    )
                                ),
                                0.7,
                                1 / (2π * model_function.delta_t * 1e-9),
                                CONFIG["nepsf"] * 8.57e-9,
                            )
                        ]
                    )...,
                    ExponentialPulse(),
                )
            )
        ),
        (
            CONFIG["rmeasurement"] == "nothing"
            ? nothing
            : (
                CONFIG["rmeasurement"] == "tomography"
                    ? SingleShotTomography(
                        N_CTRLS,
                        (
                            CONFIG["model"] == "qd1"
                            ? 2
                            : (
                                CONFIG["model"] == "qd2"
                                ? 6
                                : throw(ErrorException("Invalid model."))
                            )
                        ),
                        false,
                        false,
                    )
                    : throw(ErrorException("Invalid `rmeasurement`."))
            )
        ),
        (
            CONFIG["loss"] == "inf"
            ? gate_infidelity
            : (
                CONFIG["loss"] == "on"
                ? operator_norm
                : throw(ErrorException("Invalid loss."))
            )
        ),
        Dict(
            "logmean" => (-) ∘ log10 ∘ mean,
            "mean" => (-) ∘ mean,
            "logmax" => (-) ∘ log10 ∘ maximum,
            "max" => (-) ∘ maximum,
        )[CONFIG["statfn"]],
        (
            CONFIG["model"] == "qd1"
            ? nothing
            : (
                CONFIG["model"] == "qd2"
                ? UnitRange(2, 5)
                : throw(ErrorException("Invalid model."))
            )
        ),
        CONFIG["mapunitary"];
        n_runs=CONFIG["nruns"],
    )
elseif CONFIG["reward"] == "robustg"
        reward_function = RobustGateRewardG(
        TARGET,
        (
            CONFIG["model"] == "qd1"
            ? Simple1DSystem(
                model_function.delta_t;
                j_0=1.0 / CONFIG["tscale"],
                b=2.09 / CONFIG["tscale"],
                sigma_b=CONFIG["ndrift"] * 0.0105,
            )
            : (
                CONFIG["model"] == "qd2"
                ? QuantumDot2(
                    model_function.delta_t;
                    j_0=1.0 / CONFIG["tscale"],
                    b_ij=[1.0, 7.0, -1.0] ./ CONFIG["tscale"],
                    sigma_b=CONFIG["ndrift"] * 0.0105,
                )
                : throw(ErrorException("Invalid model."))
            )
        ),
        shaping_function.shaped_pulse_history,
        (
            CONFIG["directp"]
            ? IdentityPulse()
            : (
                CONFIG["pulse"] == "none"
                ? ExponentialPulse()
                : Chain(
                    (
                        iszero(CONFIG["nepss"])
                        ? []
                        : [
                            StaticNoiseInjection(
                                N_CTRLS, CONFIG["nepss"] * 0.0294
                            )
                        ]
                    )...,
                    (
                        iszero(CONFIG["nepsf"])
                        ? []
                        : [
                            ColouredNoiseInjection(
                                N_CTRLS,
                                (
                                    CONFIG["shaping"] == "none"
                                    ? CONFIG["inputs"]
                                    : (
                                        (
                                            CONFIG["inputs"]
                                            + shaping_function.boundary_padding[
                                                2
                                            ]
                                        )
                                        * CONFIG["srate"]
                                    )
                                ),
                                0.7,
                                1 / (2π * model_function.delta_t * 1e-9),
                                CONFIG["nepsf"] * 8.57e-9,
                            )
                        ]
                    )...,
                    ExponentialPulse(),
                )
            )
        ),
        (
            CONFIG["rmeasurement"] == "nothing"
            ? nothing
            : (
                CONFIG["rmeasurement"] == "tomography"
                    ? SingleShotTomography(
                        N_CTRLS,
                        (
                            CONFIG["model"] == "qd1"
                            ? 2
                            : (
                                CONFIG["model"] == "qd2"
                                ? 6
                                : throw(ErrorException("Invalid model."))
                            )
                        ),
                        false,
                        false,
                    )
                    : throw(ErrorException("Invalid `rmeasurement`."))
            )
        ),
        (
            CONFIG["loss"] == "inf"
            ? gate_infidelity
            : (
                CONFIG["loss"] == "on"
                ? operator_norm
                : throw(ErrorException("Invalid loss."))
            )
        ),
        Dict(
            "logmean" => (-) ∘ log10 ∘ mean,
            "mean" => (-) ∘ mean,
            "logmax" => (-) ∘ log10 ∘ maximum,
            "max" => (-) ∘ maximum,
        )[CONFIG["statfn"]],
        (
            CONFIG["model"] == "qd1"
            ? nothing
            : (
                CONFIG["model"] == "qd2"
                ? UnitRange(2, 5)
                : throw(ErrorException("Invalid model."))
            )
        ),
        CONFIG["mapunitary"],
        float(CONFIG["grscale"]);
        n_runs=CONFIG["nruns"],
    )
end
if CONFIG["normalreward"]
    reward_function = NormalisedReward(reward_function, 0.99)
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
    activation=CONFIG["activation"] == "relu" ? relu : gelu,
    init=CONFIG["init"] == "glu" ? glorot_uniform : glorot_normal,
    capacity=100000,
    hiddens=[512, 512],
    log_var_min=-15,
    log_var_max=CONFIG["logvarmax"],
    use_tqc=CONFIG["tqc"],
    n_q=25,
    k_q=46,
    dropout=CONFIG["dropout"],
    layer_norm=CONFIG["layernorm"],
    gamma=0.99,
    minibatch_size=256,
    training_steps=20,
    decays=[0.0, 0.0, 0.0, 0.0],
    clips=[5.0, 5.0, 5.0, 5.0],
    eta=CONFIG["lr"] .* ones(4),
    rho=0.005,
    warmup_normalisation_episodes=CONFIG["initnormalepisodes"],
    warmup_evaluation_episodes=CONFIG["initevalepisodes"],
    episodes=CONFIG["episodes"],
)
learn!(agent, env, CONFIG["save_directory"], CONFIG["savesteps"])
# learn!(
#     agent,
#     env,
#     "examples/quantum_dot/" * CONFIG["save_directory"],
#     CONFIG["savesteps"],
# )
