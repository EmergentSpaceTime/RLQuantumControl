using LinearAlgebra.BLAS: get_num_threads, set_num_threads

set_num_threads(Base.Threads.nthreads())
println(get_num_threads(), Base.Threads.nthreads())

using Random: seed!
using Dierckx: Spline1D
using Flux: relu, gelu, glorot_uniform, glorot_normal

using DelimitedFiles: readdlm
using TOML: parsefile
using HDF5: h5open, create_dataset, write, close
using BSON: @save, @load

include("../../src/RLQuantumControl.jl")
using .RLQuantumControl

#################
# Setup config. #
#################
CONFIG = parsefile(ARGS[1])
EPISODES = ARGS[2]
SEED = CONFIG["seed"]

# Setup seed and custom labels.
seed!(SEED)

######################
# Setup environment. #
######################
# Input function.
input_function = IdentityInput(
    3; control_min=fill(-5.4, 3), control_max=fill(2.4, 3)
)
# Shaping function.
if CONFIG["shaping"] == "none"
    shaping_function = IdentityShaping(3, CONFIG["inputs"])
elseif CONFIG["shaping"] == "fir"
    response_data = readdlm("response_data_new.txt")
    response_data[:, 2] ./= (
        maximum(abs.(response_data[:, 2]))
        * CONFIG["srate"]
        * CONFIG["filteramp"]
    )
    shaping_function = FilterShaping(
        3,
        CONFIG["inputs"],
        Spline1D(response_data[:, 1], response_data[:, 2]; bc="zero");
        boundary_values=hcat(fill(-5.4, 3), fill(-5.4, 3)),
        boundary_padding=[5, 4],
        sampling_rate=CONFIG["srate"],
    )
elseif CONFIG["shaping"] == "gauss"
    mu = CONFIG["mu"]
    sigma = CONFIG["sigma"]
    scale = 0.071 / CONFIG["srate"] * 10
    shaping_function = FilterShaping(
        3,
        CONFIG["inputs"],
        Spline1D(
            -5:0.1:5,
            @. (
                scale * exp(-0.5 * (((-5:0.1:5) - mu) / sigma) ^ 2)
                / (sigma * sqrt(2π))
            );
            bc="zero",
        );
        boundary_values=hcat(fill(-5.4, 3), fill(-5.4, 3)),
        boundary_padding=[5, 4],
        sampling_rate=CONFIG["srate"],
    )
end
# Model function.
model_function = QuantumDot2(
    ;
    delta_t=(
        CONFIG["plength"]
        / (
            hasfield(typeof(shaping_function), :shaped_pulse_history) ?
            size(shaping_function.shaped_pulse_history, 2) :
            size(shaping_function.pulse_history, 2)
        )
    ),
    sigma_b=(
        CONFIG["reward"] == "robust" ? 0.0 : CONFIG["ndrift"] * 0.0105
    ),
)
# Pulse function.
if CONFIG["pulse"] == "none"
    pulse_function = ExponentialPulse()
elseif CONFIG["pulse"] == "both"
    if CONFIG["reward"] != "robust"
        pulse_function = Chain(
            StaticNoiseInjection(3, CONFIG["nepss"] * 0.0294),
            ColouredNoiseInjection(
                3,
                (
                    CONFIG["shaping"] == "none" ?
                    CONFIG["inputs"] :
                    (
                        (
                            CONFIG["inputs"]
                            + shaping_function.boundary_padding[2]
                        )
                        * CONFIG["srate"]
                    )
                ),
                CONFIG["nepsf"] * 8.57e-9,
                0.7,
                1 / (2π * model_function.delta_t * 1e-9),
            ),
            ExponentialPulse(),
        )
    else
        pulse_function = ExponentialPulse()
    end
end
# Observation function.
if CONFIG["observation"] == "full"
    observation_function = FullObservation()
elseif CONFIG["observation"] == "noisy"
    if CONFIG["nmeasures"] == "nothing"
        observation_function = UnitaryTomography(
            3, "unitary", 6, true; n=CONFIG["nmeasures"]
        )
    else
        observation_function = FullObservation()
    end
elseif CONFIG["observation"] == "process"
    observation_function = ExactTomography(3, "process", 6, false)
end
if CONFIG["normalobs"]
    observation_function = NormalisedObservation(
        observation_function,
        isa(observation_function, FullObservation) ? 76 : 73,
    )
end
# Reward function.
if CONFIG["reward"] == "sparse"
    reward_function = SparseGateFidelity(
        [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0], 2:5
    )
elseif CONFIG["reward"] == "robust"
    reward_function = RobustGateFidelity(
        [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0],
        QuantumDot2(
            ;
            delta_t=model_function.delta_t,
            sigma_b=CONFIG["ndrift"] * 0.0105,
        ),
        shaping_function.shaped_pulse_history,
        Chain(
            StaticNoiseInjection(3, CONFIG["nepss"] * 0.0294),
            ColouredNoiseInjection(
                3,
                (
                    CONFIG["shaping"] == "none" ?
                    CONFIG["inputs"] :
                    (
                        (
                            CONFIG["inputs"]
                            + shaping_function.boundary_padding[2]
                        )
                        * CONFIG["srate"]
                    )
                ),
                CONFIG["nepsf"] * 8.57e-9,
                0.7,
                1 / (2π * model_function.delta_t * 1e-9),
            ),
            ExponentialPulse(),
        ),
        (
            CONFIG["nmeasures"] == "nothing" ?
            nothing :
            UnitaryTomography(3, "unitary", 6, true; n=CONFIG["nmeasures"])
        ),
        2:5,
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
    activation=CONFIG["activaton"] == "relu" ? relu : gelu,
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
    warmup_normalisation_episodes=150,
    warmup_evaluation_episodes=150,
    episodes=parse(Int, EPISODES),
)
r, l = learn!(agent, env)

@save CONFIG["save_directory"] * "environment.bson" env
@save CONFIG["save_directory"] * "agent.bson" agent

d_file = h5open(CONFIG["save_directory"] * "data=0.h5", "cw")
fset = create_dataset(d_file, "r", eltype(r), size(r))
fset = create_dataset(d_file, "l", eltype(l), size(l))
write(d_file["r"], r)
write(d_file["l"], l)
close(d_file)
