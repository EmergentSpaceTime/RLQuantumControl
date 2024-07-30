"""Abstract callable struct that convolves an input pulse from the input
function. Custom shapings should include a `pulse_history` field and may also
include a `shaped_pulse_history` field and a [`reset!`]() method if required.

These callables have the argument signature:
```math
    \\mathscr{S}(t, \\epsilon_{t})\\rightarrow(\\bar{t}, \\bar{\\epsilon}_{t})
```
"""
abstract type ShapingFunction <: Function end

function reset!(s::ShapingFunction)
    s.pulse_history .= zero(s.pulse_history)
    return nothing
end


struct IdentityShaping <: ShapingFunction
    pulse_history::Matrix{Float64}
end

"""
    IdentityShaping(n_controls::Int, n_inputs::Int)

Identity shaping callable that does not change the piece-wise pulses.
```math
    \\mathscr{S}(t, \\epsilon_{t}) = (t, \\epsilon_{t})
```

Args:
  * `n_controls`: Number of controls.
  * `n_inputs`: Number of inputs (corresponding to number of actions).

Fields:
  * `pulse_history`: History of controls as a (`n_controls`, `n_inputs`) matrix.
"""
function IdentityShaping(n_controls::Int, n_inputs::Int)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    n_inputs < 1 && throw(ArgumentError("`n_inputs` must be >= 1."))
    return IdentityShaping(zeros(n_controls, n_inputs))
end

function reset!(s::IdentityShaping)
    s.pulse_history .= zero(s.pulse_history)
    return nothing
end

function (s::IdentityShaping)(t_step::Int, epsilon_t::AbstractVector{Float64})
    s.pulse_history[:, t_step] = epsilon_t
    return t_step, s.pulse_history[:, t_step]
end


struct FilterShaping{T <: Union{Nothing, Matrix{Float64}}} <: ShapingFunction
    pulse_history::Matrix{Float64}
    shaped_pulse_history::Matrix{Float64}
    sampling_rate::Int
    boundary_values::T
    kernel::Spline1D
end

"""
    FilterShaping(
        n_controls::Int,
        n_inputs::Int,
        kernel::Spline1D;
        sampling_rate::Int = 10,
        boundary_values::Union{Nothing, Matrix{Float64}} = nothing,
    )

Callable for convolution with a user-defined kernel given as a `Spline1D`
object:
```math
    \\mathscr{S}(t, \\epsilon_{t}) = \\left(
        [t - 1, \\ldots, t],
        \\sum^{t}_{n}\\epsilon_{n}\\mathscr{K}[t - n]
    \\right)
```

Args:
  * `n_controls`: Number of controls.
  * `n_inputs`: Number of inputs (corresponding to number of actions).
  * `kernel`: Kernel that is convoluted with input pulses given as `Spline1D`
        object.

Kwargs:
  * `sampling_rate`: The (over-)sampling rate, i.e. the number of sub-steps per
        discrete piece-wise pulse step (default: `10`).
  * `boundary_values`: Desired boundary conditions on pulses, given as a
        (`n_controls`, `2`) matrix of beginnings and ends. The inputs are then
        appended by five values with the boundary values to ensure the shaped
        pulse ends at the boundary values. If not specified, boundary defaults
        to zeros (default: `nothing`).

Fields:
  * `pulse_history`: History of input pulses.
  * `shaped_pulse_history`: History of shaped pulses.
  * `sampling_rate`: Number of sub-steps per discrete piece-wise pulse step.
  * `boundary_values`: Desired boundary conditions on pulses.
  * `kernel`: Kernel that is convoluted with input pulses.
"""
function FilterShaping(
    n_controls::Int,
    n_inputs::Int,
    kernel::Spline1D;
    sampling_rate::Int = 10,
    boundary_values::Union{Nothing, Matrix{Float64}} = nothing,
)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    n_inputs < 1 && throw(ArgumentError("`n_inputs` must be >= 1."))
    sampling_rate < 1 && throw(ArgumentError("`sampling_rate` must be >= 1."))
    if !isnothing(boundary_values)
        if size(boundary_values) != (n_controls, 2)
            throw(
                ArgumentError(
                    "`boundary_values` must be a (`n_controls`, 2) matrix."
                )
            )
        end
    end

    pulse_history = zeros(
        n_controls,
        (n_inputs + 10 * !isnothing(boundary_values)) * sampling_rate,
    )
    if !isnothing(boundary_values)
        pulse_history[:, 1 : 5 * sampling_rate] .= boundary_values[:, 1]
        pulse_history[:, end - 5 * sampling_rate + 1 : end] .= (
            boundary_values[:, 2]
        )
    end
    return FilterShaping(
        pulse_history,
        zeros(
            n_controls,
            (n_inputs + 5 * !isnothing(boundary_values)) * sampling_rate,
        ),
        sampling_rate,
        boundary_values,
        kernel,
    )
end

function reset!(s::FilterShaping{Nothing})
    s.pulse_history .= zero(s.pulse_history)
    s.shaped_pulse_history .= zero(s.shaped_pulse_history)
    return nothing
end

function reset!(s::FilterShaping{Matrix{Float64}})
    s.pulse_history[:, 5 * s.sampling_rate + 1 : end - 5 * s.sampling_rate] .= (
        zero(Float64)
    )
    s.shaped_pulse_history .= zero(s.shaped_pulse_history)
    return nothing
end

function (s::FilterShaping{Nothing})(
    t_step::Int, epsilon_t::AbstractVector{Float64}, dt::Float64
)
    t_sub = range(s.sampling_rate * (t_step - 1) + 1, s.sampling_rate * t_step)
    s.pulse_history[:, t_sub] .= epsilon_t
    for i in t_sub
        for n in 1 : t_step * s.sampling_rate
            s.shaped_pulse_history[:, i] += (
                s.pulse_history[:, n]
                * s.kernel((i - n) * dt) * dt
            )
        end
    end
    return t_sub, s.shaped_pulse_history[:, t_sub]
end

function (s::FilterShaping{Matrix{Float64}})(
    t_step::Int, epsilon_t::AbstractVector{Float64}, dt::Float64
)
    if t_step * 10 < (size(s.pulse_history, 2) - 10 * s.sampling_rate)
        t_sub = range(
            s.sampling_rate * (t_step + 4) + 1, s.sampling_rate * (t_step + 5)
        )
        s.pulse_history[:, t_sub] .= epsilon_t
        for i in t_sub
            t = i - 5 * s.sampling_rate
            for n in 1 : (t_step + 5) * s.sampling_rate
                s.shaped_pulse_history[:, t] += (
                    s.pulse_history[:, n]
                    * s.kernel((i - n) * dt) * dt
                )
            end
        end
        return (
            t_sub .- 5 * s.sampling_rate,
            s.shaped_pulse_history[:, t_sub .- 5 * s.sampling_rate],
        )
    end
    t_sub = range(
        s.sampling_rate * (t_step + 4) + 1, size(s.pulse_history, 2)
    )
    s.pulse_history[:, t_sub[1:s.sampling_rate]] .= epsilon_t
    for i in t_sub
        t = i - 5 * s.sampling_rate
        for n in 1 : (t_step + 10) * s.sampling_rate
            s.shaped_pulse_history[:, t] += (
                s.pulse_history[:, n]
                * s.kernel((i - n) * dt) * dt
            )
        end
    end
    return (
        t_sub .- 5 * s.sampling_rate,
        s.shaped_pulse_history[:, t_sub .- 5 * s.sampling_rate],
    )
end


_n_ctrls(s::ShapingFunction) = size(s.pulse_history, 1)


_n_inpts(s::IdentityShaping) = size(s.pulse_history, 2)
_n_inpts(s::FilterShaping{Nothing}) = size(s.pulse_history, 2) ÷ s.sampling_rate

function _n_inpts(s::FilterShaping{Matrix{Float64}})
    return (size(s.pulse_history, 2) - s.sampling_rate * 10) ÷ s.sampling_rate
end


_n_ts(s::IdentityShaping) = size(s.pulse_history, 2)
_n_ts(s::FilterShaping) = size(s.shaped_pulse_history, 2)
