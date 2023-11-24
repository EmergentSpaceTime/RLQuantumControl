"""Quantum control environment evolution."""

"""Quantum control environment.

Struct of a quantum control environment. The evolution depends on the chosen
model function, input function, shaping function, pulse function(s), observation
function, and reward function. The action space can be continuous or discrete.

Kwargs:
  * model_function: Quantum model function.
  * input_function: Input function.
  * shaping_function: Shaping function.
  * pulse_function: Pulse (amplitude) function(s).
  * observation_function: Observation function that dicates the observation that
        the agent recieves at each time step.
  * reward_functon: Reward function that dicates the reward that the agent
        recieves at each time step.
  * continuous: Whether to to use continuous actions (default: true).
  * rng: Random number generator (default: default_rng()).

Fields:
  * rng: Random number generator.
  * action_space: Allowed actions on the environment.
  * model_function: The paramters and Hamiltonian components of the control
        environment as well as an evolution step.
  * input_function: Input change function.
  * shaping_function: Pulse shaping function.
  * pulse_function: Pulse (amplitude and noise) function(s).
  * observation_function: Observation function.
  * observation_space: Space of possible state elements of the environment.
  * reward_function: Reward function.
  * reward_space: Space of reward functions.
"""
struct QuantumControlEnvironment{
    R <: AbstractRNG,
    𝔸 <: AbstractVector,
    ℳ <: ModelFunction,
    ℐ <: InputFunction,
    𝒮 <: ShapingFunction,
    𝒫 <: Union{Nothing, Function, Chain},
    𝒪 <: ObservationFunction,
    ℛ <: RewardFunction,
}
    rng::R
    action_space::𝔸
    model_function::ℳ
    input_function::ℐ
    shaping_function::𝒮
    pulse_function::𝒫
    observation_function::𝒪
    observation_space::Vector{ClosedInterval{Float64}}
    reward_function::ℛ
    reward_space::ClosedInterval{Float64}
    _state::Vector{Float64}
    _stateₜ::SubArray{Float64, 0, Vector{Float64}, Tuple{Int64}, true}
    _stateₚ::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
    _stateₘ::ReshapedArray{
        Complex{Float64},
        2,
        ReinterpretArray{
            Complex{Float64},
            1,
            Float64,
            SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true},
            false,
        },
        Tuple{},
    }
    _stateₗ::SubArray{
        Complex{Float64},
        2,
        ReshapedArray{
            Complex{Float64},
            2,
            ReinterpretArray{
                Complex{Float64},
                1,
                Float64,
                SubArray{
                    Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true
                },
                false,
            },
            Tuple{},
        },
        Tuple{UnitRange{Int}, UnitRange{Int}},
        false,
    }
end

function QuantumControlEnvironment(
    ;
    model_function::ModelFunction,
    input_function::InputFunction,
    shaping_function::ShapingFunction,
    pulse_function::Union{
        Nothing, PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}
    },
    observation_function::ObservationFunction,
    reward_function::RewardFunction,
    rng::AbstractRNG = default_rng(),
    continuous::Bool = true,
)
    if isnothing(pulse_function)
        lₚ = []
    else
        lₚ = length(pulse_function)
        if isnothing(lₚ)
            lₚ = []
        end
    end
    if !allequal(
        [
            length(model_function.H_controls),
            input_function.ϵₙ,
            size(shaping_function.pulse_history, 1),
            lₚ...,
        ]
    )
        throw(
            ArgumentError(
                "Number of control Hamiltonians must be equal to number of"
                * " controls given in input function, the shaping function, and"
                * " if included, the noise function parameters in the pulse"
                * " functions."
            )
        )
    end
    if model_function.tₙ != size(shaping_function.pulse_history, 2)
        throw(
            ArgumentError(
                "Number of time steps must be equal to the number of time steps"
                * " in the shaping function."
            )
        )
    end

    if continuous
        if input_function isa StepInput
            action_space = ClosedInterval.(
                -input_function.Δϵ, input_function.Δϵ
            )
        else
            action_space = ClosedInterval.(
                input_function.ϵₘᵢₙ, input_function.ϵₘₐₓ
            )
        end
    else
        action_space = OneTo(
            (2 + (InputFunction <: StepInput)) ^ input_function.ϵₙ
        )
    end

    _state = vcat(
        model_function.tₙ,
        zeros(Float64, input_function.ϵₙ),
        copy(
            vec(
                reinterpret(
                    Float64,
                    Matrix{Complex{Float64}}(
                        I,
                        size(model_function(rand(Float64, input_function.ϵₙ))),
                    ),
                )
            )
        ),
    )
    _stateₜ = view(_state, 1)
    _stateₚ = view(_state, 2 : 1 + input_function.ϵₙ)
    _stateₘ = reshape(
        reinterpret(
            Complex{Float64}, @view _state[2 + input_function.ϵₙ : end]
        ),
        isqrt((length(_state) - 1 - input_function.ϵₙ) ÷ 2),
        isqrt((length(_state) - 1 - input_function.ϵₙ) ÷ 2),
    )
    _stateₗ = @view _stateₘ[
        model_function._computational_indices,
        model_function._computational_indices,
    ]

    _state_space = vcat(
        ClosedInterval(0.0, convert(Float64, model_function.tₙ)),
        ClosedInterval.(input_function.ϵₘᵢₙ, input_function.ϵₘₐₓ),
        ClosedInterval.(-ones(2 * length(_stateₘ)), ones(2 * length(_stateₘ))),
    )
    return QuantumControlEnvironment(
        rng,
        action_space,
        model_function,
        input_function,
        shaping_function,
        pulse_function,
        observation_function,
        observation_function(_state_space),
        reward_function,
        reward_space(reward_function),
        _state,
        _stateₜ,
        _stateₚ,
        _stateₘ,
        _stateₗ,
    )
end

##################
# Public Methods #
##################
"""Get the pulse history of an environment."""

"""Environment reset function.

Args:
  * env: Quantum control environment.

Returns:
  * Vector{Float64}: An initial environment observation.
"""
function reset!(env::QuantumControlEnvironment)
    reset!(env.model_function, env.rng)
    reset!(env.shaping_function)
    if !isnothing(env.pulse_function)
        reset!(env.pulse_function, env.rng)
    end

    env._stateₜ .= env.model_function.tₙ
    env._stateₚ .= zero(env._stateₚ)
    env._stateₘ .= Matrix(I, size(env._stateₘ))
    return env.observation_function(env._state)
end

"""Environment step function.

Input a valid action to take a step in the environment modifying it's state and
getting an observation, reward, and a termination if ended.

Args:
  * env: Quantum control environment.
  * action: Chosen action that the agent takes.

Returns:
  * Tuple: A tuple containing:
    - observation (Vector{Float64}): An observation of the quantum control
        environment.
    - done (Bool): Indicates if the environment has terminated.
    - reward (Float64): The reward recieved.
"""
function step!(
    env::QuantumControlEnvironment, action::Union{Vector{Float64}, Int}
)
    if !_is_valid_action(env, action)
        throw(DomainError(action, "Action is not valid."))
    end

    env._stateₜ .-= 1
    env._stateₚ .= env.input_function(env._stateₚ, action)

    ϵ̄ₜ = env.shaping_function(env._stateₚ)
    for t in axes(ϵ̄ₜ, 2)
        if isnothing(env.pulse_function)
            U = env.model_function(ϵ̄ₜ[:, t])
        else
            U = env.model_function(env.pulse_function(ϵ̄ₜ[:, t]))
        end
        # env._stateₘ .= matmul(U, env._stateₘ)
        # matmul!(env._stateₘ, U, env._stateₘ)
        env._stateₘ .= U * env._stateₘ
        if !is_unitary(env._stateₘ)
            throw(DomainError(env._stateₘ, "State is not unitary."))
        end
    end
    done = env._state[1] <= 0
    reward = env.reward_function(env._stateₗ, done)
    observation = env.observation_function(env._state)
    return observation, done, reward
end

_is_valid_action(env::QuantumControlEnvironment, a::Int) = a ∈ env.action_space

function _is_valid_action(env::QuantumControlEnvironment, a::Vector{Float64})
    for i in eachindex(a)
        (a[i] - 1e-5 * sign(a[i])) ∈ env.action_space[i] || return false
    end
    return true
end
