"""Contains reward functions for the quantum control environment."""


abstract type RewardFunction <: Function end


"""Dense negative log-infidelity reward.

Constructs a fidelity reward function that is defined as:

```math
ℛ(unitaryₜ) = rₜ - rₜ₋₁ where rₜ = -log₁₀(1 - ℱ(unitaryₜ, target))
```

Args:
  * target: Target unitary matrix.

Fields:
  * target: Target unitary matrix.
"""
struct DenseFidelityReward{M <: Number} <: RewardFunction
    target::Matrix{M}
    rₜ₋₁::RefValue{Float64}
end

function DenseFidelityReward(target::Matrix{<:Number})
    if !is_unitary(target)
        throw(ArgumentError("Target matrix must be unitary!"))
    end
    return DenseFidelityReward(target, RefValue(zero(Float64)))
end

reward_space(::DenseFidelityReward) = ClosedInterval(0, -log10(eps(Float64)))

function (r::DenseFidelityReward)(
    unitary::AbstractMatrix{ComplexF64}, done::Bool
)
    nlifₜ = -log10(1 - gate_fidelity(unitary, r.target) + 10 * eps(Float64))
    rₜ = nlifₜ - r.rₜ₋₁[]
    r.rₜ₋₁[] = nlifₜ
    if done
        r.rₜ₋₁[] = 0
    end
    return rₜ
end


"""Sparse negative log-infidelity reward.

Constructs a fidelity reward function that is defined as:

```math
ℛ(unitaryₜ) = 0 ∀ t ≠ T & -log₁₀(1 - ℱ(unitaryₜ, target)) for t == T
```

Args:
  * target: Target unitary matrix.

Fields:
  * target: Target unitary matrix.
"""
struct SparseFidelityReward{M <: Number} <: RewardFunction
    target::Matrix{M}
end

function SparseFidelityReward(target::Matrix{<:Number})
    if !is_unitary(target)
        throw(ArgumentError("Target matrix must be unitary!"))
    end
    return SparseFidelityReward{eltype(target)}(target)
end

reward_space(::SparseFidelityReward) = ClosedInterval(0, -log10(eps(Float64)))

function (r::SparseFidelityReward)(
    unitary::AbstractMatrix{ComplexF64}, done::Bool
)
    if done
        return -log10(1 - gate_fidelity(unitary, r.target) + 10 * eps(Float64))
    else
        return zero(Float64)
    end
end

# struct LeakageReward <: RewardFunction end

# @inline function (::LeakageReward)(
#     unitary::AbstractMatrix{Complex{T}},
#     target::AbstractMatrix{Complex{T}},
#     done::Bool,
# ) where {T <: AbstractFloat}
#     if done
#         ℒ = abs(1 - tr(matmul(unitary, unitary')) / size(unitary, 1))
#         r₁ = -ℒ
#         # r₁ = -log10(ℒ + eps(T))
#         F = svd(unitary)
#         Δ = norm(matmul(F.U, F.Vt) .- target)
#         r₂ = -Δ
#         # Δ = 1 - gate_fidelity(matmul(F.U, F.Vt), target)
#         # r₂ = -log10(Δ + eps(T))
#         return r₁ + r₂
#     else
#         return zero(T)
#     end
# end

"""Reward function with return values normalised to 𝒩(0, 1).

Args:
  * base_function: Base reward function.
  * γ: Discount factor.

Fields:
  * base_function: Base reward function.
  * γ: Discount factor.
  * returns_e: Return of episode.
  * returns_μ: Return mean.
  * return_σ²: Return variance.
  * count: Number of observed returns.
"""
struct NormalisedReward{ℛ <: RewardFunction} <: RewardFunction
    base_function::ℛ
    γ::Float64
    return_e::RefValue{Float64}
    return_μ::RefValue{Float64}
    return_σ²::RefValue{Float64}
    count::RefValue{Int}
end

function NormalisedReward(base_function::RewardFunction, γ::Float64)
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
        γ,
        RefValue(zero(Float64)),
        RefValue(zero(Float64)),
        RefValue(one(Float64)),
        RefValue(1),
    )
end

reward_space(r::NormalisedReward) = reward_space(r.base_function)

function (r::NormalisedReward)(unitary::AbstractMatrix{ComplexF64}, done::Bool)
    reward = r.base_function(unitary, done)
    r.return_e[] = r.return_e[] * r.γ + reward
    Δᵣ = r.return_e[] - r.return_μ[]
    r.return_μ[] += Δᵣ / (r.count[] + 1)
    r.return_σ²[] = (
        r.return_σ²[] * r.count[] / (r.count[] + 1)
        + (Δᵣ ^ 2) * r.count[] / (r.count[] + 1) ^ 2
    )
    r.count[] += 1
    if done
        r.return_e[] = 0
    end
    return reward, reward / sqrt(r.return_σ²[] + eps(Float64))
end
