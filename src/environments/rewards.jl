"""Abstract callable struct to generate different types of rewards from the
quantum simulation. Custom rewards should define a [`reward_space`]() method.

These callables have the argument signature:
```math
    \\mathscr{R}(s_{t}, done_{t})
```
"""
abstract type RewardFunction <: Function end


struct DenseGateFidelity{M <: Number} <: RewardFunction
    target::Matrix{M}
    _r_tm1::Base.RefValue{Float64}
end

"""
    DenseGateFidelity(target::Matrix)

Callable to generate a gate fidelity reward function that is defined as:
```math
    \\mathscr{R}(U_{t}) = r_{t} - r_{t - 1}
```
Where:
```math
    r_{t} = -\\log_{10}(1 - \\mathcal{F}(U_{t}, U_{\\text{target}}))
```

Args:
  * `target`: Target unitary matrix.

Fields:
  * `target`: Target unitary matrix.
"""
function DenseGateFidelity(target::Matrix)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    return DenseGateFidelity(target, Base.RefValue(zero(Float64)))
end

function (r::DenseGateFidelity)(u::AbstractMatrix{ComplexF64}, done::Bool)
    nlif = -log10(1 - gate_fidelity(u, r.target) + 1e-6)
    r = nlif - r._r_tm1[]
    r._r_tm1[] = nlif
    if done
        r._r_tm1[] = 0
    end
    return r
end

reward_space(::DenseGateFidelity) = ClosedInterval(0.0, 6.0)


struct SparseGateFidelity{M <: Number} <: RewardFunction
    target::Matrix{M}
end

"""
    SparseGateFidelity(target::Matrix)

Constructs a fidelity reward function that is defined as:

```math
    \\mathscr{R}(U_{t}) = \\begin{cases}
        0\\ \\forall\\ t \\neq T\\\\
        r_{t}\\ \\ \\ t = T
    \\end{cases}
```
Where:
```math
    r_{T} = -\\log_{10}(1 - \\mathcal{F}(U_{T}, U_{\\text{target}}))
```

Args:
  * `target`: Target unitary matrix.

Fields:
  * `target`: Target unitary matrix.
"""
function SparseGateFidelity(target::Matrix)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    return SparseGateFidelity{eltype(target)}(target)
end

function (r::SparseGateFidelity)(u::AbstractMatrix{ComplexF64}, done::Bool)
    if done
        return -log10(1 - gate_fidelity(u, r.target) + 1e-6)
    end
    return zero(Float64)
end

reward_space(::SparseGateFidelity) = ClosedInterval(0.0, 6.0)


struct RobustGateFidelity{
    T <: Number,
    M <: ModelFunction,
    P <: Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}},
} <: RewardFunction
    target::Matrix{T}
    model_function::M
    pulse_function::P
    n_runs::Int
    _pulse_history::SubArray{
        Float64,
        2,
        Matrix{Float64},
        Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}},
        true,
    }
end

"""
    RobustGateFidelity(
        target::Matrix{ComplexF64},
        model_function::ModelFunction,
        pulse_history::Matrix{Float64},
        pulse_function::Union{
            PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}
        };
        n_runs::Int = 50,
    )
"""
function RobustGateFidelity(
    target::Matrix{<:Number},
    model_function::ModelFunction,
    pulse_history::Matrix{Float64},
    pulse_function::Union{PulseFunction, Chain{<:Tuple{Vararg{PulseFunction}}}};
    n_runs::Int = 50,
)
    is_unitary(target) || throw(ArgumentError("Target matrix must be unitary!"))
    if !has_noise(model_function) & !has_noise(pulse_function)
        throw(
            ArgumentError(
                "Do not use this reward if the process does not contain noise"
                * "processes."
            )
        )
    end
    return RobustGateFidelity(
        target,
        model_function,
        pulse_function,
        n_runs,
        view(pulse_history, :, :),
    )
end

function (r::RobustGateFidelity)(
    ::AbstractMatrix{ComplexF64}, done::Bool, rng::AbstractRNG = default_rng()
)
    if done
        rewards = zeros(Float64, r.n_runs)
        for i in 1:r.n_runs
            reset!(r.model_function, rng)
            reset!(r.pulse_function, rng)
            u = Matrix{ComplexF64}(I, _m_size(r.model_function))
            for i in axes(r._pulse_history, 2)
                u .= (
                    u
                    * r.model_function(
                        r.pulse_function(i, r._pulse_history[:, i])
                    )
                )
            end
            rewards[i] = -log10(
                1
                - gate_fidelity(
                    u[
                        computational_indices(r.model_function),
                        computational_indices(r.model_function),
                    ],
                    r.target,
                )
                + 1e-6
            )
        end
        return mean(rewards)
    end
    return zero(Float64)
end

reward_space(::RobustGateFidelity) = ClosedInterval(0.0, 6.0)


struct NormalisedReward{R <: RewardFunction} <: RewardFunction
    base_function::R
    gamma::Float64
    return_e::Base.RefValue{Float64}
    returns_mean::Base.RefValue{Float64}
    returns_var::Base.RefValue{Float64}
    count::Base.RefValue{Int}
end

"""
    NormalisedReward(base_function::RewardFunction, gamma::Float64)

Reward function with output values normalised to unit normal.

Args:
  * `base_function`: Base reward function.
  * `gamma`: Discount factor.

Fields:
  * `base_function`: Base reward function.
  * `gamma`: Discount factor.
  * `return_e`: Return of episode.
  * `returns_mean`: Return mean.
  * `returns_var`: Return variance.
  * `count`: Number of observed returns.
"""
function NormalisedReward(base_function::RewardFunction, gamma::Float64)
    if base_function isa NormalisedReward
        throw(
            ArgumentError(
                "Base reward function can't already be the normalised reward"
                * "function!"
            )
        )
    end
    return NormalisedReward(
        base_function,
        gamma,
        Base.RefValue(zero(Float64)),
        Base.RefValue(zero(Float64)),
        Base.RefValue(one(Float64)),
        Base.RefValue(1),
    )
end

function (r::NormalisedReward)(state_m::AbstractMatrix{ComplexF64}, done::Bool)
    reward = r.base_function(state_m, done)
    r.return_e[] = r.return_e[] * r.gamma + reward
    Δᵣ = r.return_e[] - r.returns_mean[]
    r.returns_mean[] += Δᵣ / (r.count[] + 1)
    r.returns_var[] = (
        r.returns_var[] * r.count[] / (r.count[] + 1)
        + (Δᵣ ^ 2) * r.count[] / (r.count[] + 1) ^ 2
    )
    r.count[] += 1
    if done
        r.return_e[] = 0
    end
    return reward, reward / sqrt(r.returns_var[] + 1e-6)
end

reward_space(r::NormalisedReward) = reward_space(r.base_function)
