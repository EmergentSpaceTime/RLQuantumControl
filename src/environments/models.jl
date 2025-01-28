#################
# Natural Units #
#################
# 1 / ns = 6.582119569e-7 eV
# 1mT = 0.087953 GHz

"""Abstract callable struct that creates an evolution operator (e.g. unitary
from Hamiltonian) from an inputted pulse (at the that time step). Models may
have an optional [`reset!`]() method that sets an initial state for the model
(i.e. the cases where the model contains random noise or parameters which can
differ episode to episode), and a [`has_noise`]() method outputting a `Bool` if
the dynamics contains noise.

These callables have the argument signature:
```math
    \\mathscr{M}(\\epsilon_{t})\\rightarrow\\xi_{t}
```
"""
abstract type ModelFunction end

reset!(::ModelFunction, ::AbstractRNG = default_rng()) = nothing
has_noise(::ModelFunction) = false

function _n_ctrls(m::ModelFunction)
    if hasfield(typeof(m), :h_controls)
        if m.h_controls isa Matrix
            return 1
        elseif m.h_controls isa Vector
            return length(m.h_controls)
        end
    end
    throw(
        ErrorException(
            "Model doesn't have defined control fields, `h_controls`."
        )
    )
end

function _m_size(m::ModelFunction)
    if hasfield(typeof(m), :h_drifts)
        if m.h_drifts isa Matrix
            return size(m.h_drifts)
        elseif m.h_drifts isa Vector
            return size(m.h_drifts[1])
        end
    end
    throw(
        ErrorException(
            "Model doesn't have defined drift Hamiltonian fields, `h_drifts`."
        )
    )
end


struct QuantumDot2 <: ModelFunction
    delta_t::Float64
    h_drifts::Vector{Matrix{Float64}}
    h_controls::Vector{Matrix{Float64}}
    h_coupling::Matrix{Float64}
    _sigma_bb::Vector{Float64}
    _h_drift_episode::Matrix{Float64}
end

"""
    QuantumDot2(
        ;
        delta_t::Real = 1.0,
        b_ij::Vector{<:Real} = [1.0, 7.0, -1.0],
        j_0::Real = 1.0,
        epsilon_0::Real = 0.272,
        e_coupling::Real = 0.0,
        sigma_b::Real = 0.0,
    )

Two-qubit double quantum dot model with three controls
[PhysRevB.101.155311](@cite) and a Schrödinger evolution. Two spins corresponds
to one logical qubit. Units are in GHz unless specified otherwise.

```math
    \\mathscr{M}(\\epsilon_{t}) = U_{t} = {%
        \\exp\\left(-i\\Delta tH(\\epsilon_{t})\\right)
    }
```
Where:
```math
    \\begin{aligned}
        H(\\epsilon_{t}) &= \\frac{b_{12}}{8}(%
            -3\\sigma^{z(1)} + \\sigma^{z(2)} + \\sigma^{z(3)} + \\sigma^{z(4)}
        )\\\\
        &+ \\frac{b_{23}}{4}(%
            -\\sigma^{z(1)} - \\sigma^{z(2)} + \\sigma^{z(3)} + \\sigma^{z(4)}
        )\\\\
        &+ \\frac{b_{34}}{8}(%
            -\\sigma^{z(1)} - \\sigma^{z(2)} - \\sigma^{z(3)} + 3\\sigma^{z(4)}
        )\\\\
        &+ {%
            \\sum^{3}_{i}
            \\frac{%
                J_{0}
                \\exp{%
                    \\left(
                        \\frac{\\epsilon_{i, i + 1, t}}{\\epsilon_{0}}
                    \\right)
                }
            }{4}
            \\bar{\\sigma}^{(i)}\\cdot\\bar{\\sigma}^{(i + 1)}
        }
    \\end{aligned}
```
In addition one can add a quasi-static (slow) noise that peturbs the magnetic
field gradients as well as a empirical coupling term.

Kwargs:
  * `delta_t`: Time step in ns. Note that if pulse is oversampled and shaped,
        use the sub-time step (default: `1.0`).
  * `b_ij`: Magnetic field gradients between i-j qubits (default:
        `[1.0, 7.0, -1.0]`).
  * `j_0`: Unit parameter for tilt-control (default: `1.0`).
  * `epsilon_0`: Unit field in meV (default: `0.272`).
  * `e_coupling`: Coupling parameter for leakage in ``\\mu``eV (default:
        `0.0`).
  * `sigma_b`: Standard deviation of the slow noise on the magnetic field
        gradients (default: `0.0`).

Fields:
  * `h_drift`: Drift components of Hamiltonian.
  * `h_controls`: Control components of Hamiltonian.
  * `h_coupling`: Coupling component of Hamiltonian.
"""
function QuantumDot2(
    ;
    delta_t::Real = 1.0,
    b_ij::Vector{<:Real} = [1.0, 7.0, -1.0],
    j_0::Real = 1.0,
    epsilon_0::Real = 0.272,
    e_coupling::Real = 0.0,
    sigma_b::Real = 0.0,
)
    if delta_t <= zero(delta_t)
        throw(ArgumentError("`delta_t` must be positive."))
    end
    if sigma_b < zero(sigma_b)
        throw(ArgumentError("`sigma_b` must be positive."))
    end
    iszero(epsilon_0) && throw(ArgumentError("`epsilon_0` must be non-zero."))
    j_0 < zero(j_0) && throw(ArgumentError("`j_0` must be positive."))
    return QuantumDot2(
        delta_t,
        b_ij .* _get_h_drift_terms_qdot_2(),
        j_0 .* _get_H_control_terms_qdot_2(),
        _get_h_coupling_term_qdot_2(j_0, epsilon_0, e_coupling),
        iszero(sigma_b) ? [] : fill(sigma_b, 3) ./ b_ij,
        zeros(6, 6),
    )
end

function (m::QuantumDot2)(epsilon_t::Vector{Float64})
    return cis(
        -Hermitian(
            @. m.delta_t * (
                m._h_drift_episode
                + epsilon_t[1] * m.h_controls[1]
                + epsilon_t[2] * m.h_controls[2]
                + epsilon_t[3] * m.h_controls[3]
                + epsilon_t[1] * epsilon_t[3] * m.h_coupling
            )
        )
    )
end

function reset!(m::QuantumDot2, rng::AbstractRNG = default_rng())
    if isempty(m._sigma_bb)
        m._h_drift_episode .= sum(m.h_drifts)
    else
        @. m._h_drift_episode = $sum(
            m.h_drifts * (1 + $randn(rng, 3) * m._sigma_bb)
        )
    end
    return nothing
end

has_noise(m::QuantumDot2) = !isempty(m._sigma_bb)
_m_size(::QuantumDot2) = 6, 6


function _get_h_drift_terms_qdot_2()
    sigma_z = [1 0; 0 -1]
    sigma_z_1 = kron(sigma_z, I(2), I(2), I(2))
    sigma_z_2 = kron(I(2), sigma_z, I(2), I(2))
    sigma_z_3 = kron(I(2), I(2), sigma_z, I(2))
    sigma_z_4 = kron(I(2), I(2), I(2), sigma_z)

    h_12 = @. (-3 * sigma_z_1 + sigma_z_2 + sigma_z_3 + sigma_z_4) / 8
    h_23 = @. (-sigma_z_1 - sigma_z_2 + sigma_z_3 + sigma_z_4) / 4
    h_34 = @. (-sigma_z_1 - sigma_z_2 - sigma_z_3 + 3 * sigma_z_4) / 8

    indices = [4, 6, 7, 10, 11, 13]
    return [
        h_12[indices, indices], h_23[indices, indices], h_34[indices, indices]
    ]
end


function _get_H_control_terms_qdot_2()
    sigma_x, sigma_y, sigma_z = [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]

    sigma_12 = (
        kron(sigma_x, sigma_x, I(2), I(2))
        .+ real(kron(sigma_y, sigma_y, I(2), I(2)))
        .+ kron(sigma_z, sigma_z, I(2), I(2))
    ) ./ 4
    sigma_23 = (
        kron(I(2), sigma_x, sigma_x, I(2))
        .+ real(kron(I(2), sigma_y, sigma_y, I(2)))
        .+ kron(I(2), sigma_z, sigma_z, I(2))
    ) ./ 4
    sigma_34 = (
        kron(I(2), I(2), sigma_x, sigma_x)
        .+ real(kron(I(2), I(2), sigma_y, sigma_y))
        .+ kron(I(2), I(2), sigma_z, sigma_z)
    ) ./ 4

    indices = [4, 6, 7, 10, 11, 13]
    return [
        sigma_12[indices, indices],
        sigma_23[indices, indices],
        sigma_34[indices, indices],
    ]
end


function _get_h_coupling_term_qdot_2(
    j_0::Real, epsilon_0::Real, e_coupling::Real
)
    return diagm([0, 0, 0, 0, 1, 0]) .* (
        6.582119e-7 * e_coupling * (j_0 / epsilon_0) ^ 2
    )
end
