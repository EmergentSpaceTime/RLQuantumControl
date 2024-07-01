struct QuantumControlEnvironment{
    I <: InputFunction,
    S <: ShapingFunction,
    P <: Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
    M <: ModelFunction,
    O <: ObservationFunction,
    R <: RewardFunction,
}
    input_function::I
    shaping_function::S
    action_space::Vector{ClosedInterval{Float64}}
    n_controls::Int
    n_inputs::Int
    pulse_function::P
    model_function::M
    observation_function::O
    observation_space::Vector{ClosedInterval{Float64}}
    reward_function::R
    reward_space::ClosedInterval{Float64}
    _t_step::Base.RefValue{Int}
    _state::Vector{Float64}
    _state_t::SubArray{Float64, 0, Vector{Float64}, Tuple{Int64}, true}
    _state_p::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    _state_m::Base.ReshapedArray{
        Complex{Float64},
        2,
        Base.ReinterpretArray{
            Complex{Float64},
            1,
            Float64,
            SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true},
            false,
        },
        Tuple{},
    }
end

"""
    QuantumControlEnvironment(
        ;
        input_function::InputFunction,
        shaping_function::ShapingFunction,
        pulse_function::Union{
            PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}
        },
        model_function::ModelFunction,
        observation_function::ObservationFunction,
        reward_function::RewardFunction,
    )

Struct of a quantum control environment. The evolution depends on the chosen
input function, shaping function, pulse function(s), model function, observation
function, and reward function.

Kwargs:
  * `input_function`: Input function.
  * `shaping_function`: Shaping function.
  * `pulse_function`: Pulse (amplitude) function(s).
  * `model_function`: Quantum model function.
  * `observation_function`: Observation function that dicates the observation
        that the agent recieves at each time step.
  * `reward_functon`: Reward function that dicates the reward that the agent
        recieves at each time step.

Fields:
  * `input_function`: Input change function.
  * `shaping_function`: Pulse shaping function.
  * `action_space`: Allowed actions on the environment.
  * `n_controls`: Number of controls.
  * `n_inputs`: Number of inputs (corresponding to number of actions).
  * `pulse_function`: Pulse (amplitude and noise) function(s).
  * `model_function`: The paramters and Hamiltonian components of the control
        environment as well as an evolution step.
  * `observation_function`: Observation function.
  * `observation_space`: Space of possible state elements of the environment.
  * `reward_function`: Reward function.
  * `reward_space`: Space of reward functions.
"""
function QuantumControlEnvironment(
    ;
    input_function::InputFunction,
    shaping_function::ShapingFunction,
    pulse_function::Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
    model_function::ModelFunction,
    observation_function::ObservationFunction,
    reward_function::RewardFunction,
)
    if !allequal(
        [
            _n_ctrls(input_function),
            _n_ctrls(shaping_function),
            (
                isnothing(_n_ctrls(pulse_function))
                ? []
                : _n_ctrls(pulse_function)
            )...,
            _n_ctrls(model_function),
        ]
    )
        throw(
            ArgumentError(
                "Number of control Hamiltonians must be equal to"
                * "`length(input_function.control_min)`, the first axis of "
                * " `shaping_function.pulse_history`, and"
                * " `length(pulse_function)` if it contains noise injectors."
            )
        )
    end
    n_controls = size(shaping_function.pulse_history, 1)
    if !allequal(
        [
            _n_ts(shaping_function),
            (
                isnothing(_n_ts(pulse_function))
                ? []
                : _n_ts(pulse_function)
            )...,
        ]
    )
        throw(
            ArgumentError(
                "Second axis of `shaping_function.pulse_history` must be equal"
                * " to the time-step length of `pulse_function` if it contains"
                * " noise injectors."
            )
        )
    end
    n_inputs = _n_inpts(shaping_function)
    if (
        isa(reward_function, RobustGateFidelity)
        | isa(reward_function, NormalisedReward{<:RobustGateFidelity})
    )
        if has_noise(model_function) | has_noise(pulse_function)
            throw(
                ArgumentError(
                    "Use a `model_function` and `pulse_function` without noise"
                    * " in the environment if using `RobustGateFidelity`."
                )
            )
        end
    end

    _state = vcat(
        n_inputs,
        zeros(Float64, n_controls),
        copy(
            vec(
                reinterpret(
                    Float64,
                    Matrix{Complex{Float64}}(I, _m_size(model_function)),
                )
            )
        ),
    )
    _state_t = view(_state, 1)
    _state_p = view(_state, 2 : 1 + n_controls)
    _state_m = reshape(
        reinterpret(Complex{Float64}, @view _state[2 + n_controls : end]),
        isqrt((length(_state) - 1 - n_controls) ÷ 2),
        isqrt((length(_state) - 1 - n_controls) ÷ 2),
    )
    _state_space = vcat(
        ClosedInterval(0.0, convert(Float64, n_inputs)),
        ClosedInterval.(input_function.control_min, input_function.control_max),
        ClosedInterval.(
            -ones(2 * length(_state_m)), ones(2 * length(_state_m))
        ),
    )
    return QuantumControlEnvironment(
        input_function,
        shaping_function,
        action_space(input_function),
        n_controls,
        n_inputs,
        pulse_function,
        model_function,
        observation_function,
        observation_space(observation_function, _state_space),
        reward_function,
        reward_space(reward_function),
        Base.RefValue(0),
        _state,
        _state_t,
        _state_p,
        _state_m,
    )
end

"""
    reset!(env::QuantumControlEnvironment, rng::AbstractRNG = default_rng())

Reset the environment to it's initial state.

Args:
  * `env`: Quantum control environment.
  * `rng`: Random number generator (default: [`Random.default_rng()`]()).

Returns:
  * `Vector{Float64}`: An initial environment observation.
"""
function reset!(
    env::QuantumControlEnvironment, rng::AbstractRNG = default_rng()
)
    reset!(env.model_function, rng)
    reset!(env.shaping_function)
    reset!(env.pulse_function, rng)

    env._t_step[] = 0
    env._state_t .= env.n_inputs
    env._state_p .= zero(env._state_p)
    env._state_m .= Matrix(I, size(env._state_m))
    return env.observation_function(env._state)
end

"""
    step!(env::QuantumControlEnvironment, action::Vector{Float64})

Input a valid action to take a step in the environment modifying it's state and
getting an observation, reward, and a termination if ended.

Args:
  * `env`: Quantum control environment.
  * `action`: Chosen action that the agent takes.

Returns:
  * `Tuple`: A tuple containing:
    - `observation` (`Vector{Float64}`): An observation of the quantum control
        environment.
    - `done` (`Bool`): Indicates if the environment has terminated.
    - `reward` (`Float64`): The reward recieved.
"""
function step!(env::QuantumControlEnvironment, action::Vector{Float64})
    if !is_valid_input(env.action_space, action)
        throw(DomainError(action, "Action is not valid."))
    end

    env._t_step[] += 1
    env._state_t .-= 1
    env._state_p .= env.input_function(env._state_p, action)

    t_bar, epsilon_t_bar = env.shaping_function(env._t_step[], env._state_p)
    for i in axes(epsilon_t_bar, 2)
        u = env.model_function(
            env.pulse_function(t_bar[i], epsilon_t_bar[:, i])
        )
        env._state_m .= u * env._state_m
    end
    done = env._state[1] <= 0
    reward = env.reward_function(env._state_m, done)
    observation = env.observation_function(env._state)
    return observation, done, reward
end
