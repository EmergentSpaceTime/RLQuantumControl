"""Abstract callable struct that convolves an input pulse from the input
function. Custom shapings should include a `pulse_history` field and may also
include a `shaped_pulse_history` field and a [`reset!`]() method if required.
Private methods `_n_inpts` and `_n_ts` are used to determine aspects of the
shaping function and may need to be implemented for custom shapings.

These callables have the argument signature:
```math
    {%
        \\mathscr{S}(t, \\epsilon_{t}, \\delta t)
        \\rightarrow
        (\\bar{t}, \\bar{\\epsilon}_{t})
    }
```
"""
abstract type ShapingFunction <: Function end

function reset!(s::ShapingFunction)
    s.pulse_history .= zero(s.pulse_history)
    return nothing
end

_n_ctrls(s::ShapingFunction) = size(s.pulse_history, 1)


struct IdentityShaping <: ShapingFunction
    pulse_history::Matrix{Float64}
end

"""
    IdentityShaping(n_controls::Int, n_inputs::Int)

Identity shaping callable that does not change the piece-wise pulses.
```math
    \\mathscr{S}(t, \\epsilon_{t}, \\delta t) = (t, \\epsilon_{t})
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

function (s::IdentityShaping)(
    t_step::Int, epsilon_t::AbstractVector{Float64}, ::Float64)
    s.pulse_history[:, t_step] = epsilon_t
    return t_step, s.pulse_history[:, t_step]
end

_n_inpts(s::IdentityShaping) = size(s.pulse_history, 2)
_n_ts(s::IdentityShaping) = size(s.pulse_history, 2)


struct FilterShaping{
    T <: Union{Nothing, Matrix{Float64}}, S <: Union{Nothing, Vector{Int}}
} <: ShapingFunction
    pulse_history::Matrix{Float64}
    shaped_pulse_history::Matrix{Float64}
    sampling_rate::Int
    boundary_values::T
    boundary_padding::S
    kernel::Spline1D
end

"""
    FilterShaping(
        n_controls::Int,
        n_inputs::Int,
        kernel::Spline1D;
        sampling_rate::Int = 10,
        boundary_values::Union{Nothing, Matrix{Float64}} = nothing,
        boundary_padding::Union{Nothing, Vector{Int}} = nothing,
    )

Callable for convolution with a user-defined kernel given as a
`Dierckx.Spline1D` object. The output is given by:
```math
    \\mathscr{S}(t, \\epsilon_{t}, \\delta t) = \\left(
        [t - 1, \\ldots, t],
        \\sum^{t}_{n}\\epsilon_{n}\\mathscr{K}[t - n]
    \\right)
```
Where ``\\mathscr{K}`` is the kernel and ``[t - 1, \\ldots, t]`` represents the
sub-time steps when oversampling.

Args:
  * `n_controls`: Number of controls.
  * `n_inputs`: Number of inputs (corresponding to number of actions).
  * `kernel`: Kernel that is convoluted with input pulses.

Kwargs:
  * `sampling_rate`: The (over-)sampling rate, i.e. the number of sub-steps per
        discrete piece-wise pulse step (default: `10`).
  * `boundary_values`: Desired boundary conditions on pulses, given as a
        (`n_controls`, `2`) matrix of beginnings and ends. The inputs are then
        appended by `boundary_padding` values on each side with the boundary
        values to ensure the shaped pulse ends at the boundary values. If not
        specified, no boundary is included (default: `nothing`).
  * `boundary_padding`: Number of padding values to add to the beginning and end
        of the sequence (default: `nothing`).

Fields:
  * `pulse_history`: History of input pulses.
  * `shaped_pulse_history`: History of shaped pulses.
  * `sampling_rate`: Number of sub-steps per discrete piece-wise pulse step.
  * `boundary_values`: Desired boundary conditions on pulses.
  * `boundary_padding`: Number of padding values to add to the beginning and end
        of the sequence.
  * `kernel`: Kernel that is convoluted with input pulses.
"""
function FilterShaping(
    n_controls::Int,
    n_inputs::Int,
    kernel::Spline1D;
    sampling_rate::Int = 10,
    boundary_values::Union{Nothing, Matrix{Float64}} = nothing,
    boundary_padding::Union{Nothing, Vector{Int}} = nothing,
)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    n_inputs < 1 && throw(ArgumentError("`n_inputs` must be >= 1."))
    sampling_rate <= 1 && throw(ArgumentError("`sampling_rate` must be > 1."))
    if !isnothing(boundary_values)
        if isnothing(boundary_padding)
            throw(
                ArgumentError(
                    "`boundary_padding` can't be nothing if `boundary_values`"
                    * " is not nothing."
                )
            )
        end
        if size(boundary_values) != (n_controls, 2)
            throw(
                ArgumentError(
                    "`boundary_values` must be a (`n_controls`, 2) matrix."
                )
            )
        end
    end
    if !isnothing(boundary_padding)
        if length(boundary_padding) != 2
            throw(
                ArgumentError(
                    "`boundary_padding` must be a vector of length two."
                )
            )
        end
        if iszero(boundary_padding)
            throw(
                ArgumentError(
                    "`boundary_padding` elements must contain one side greater"
                    * " than 0 if using them."
                )
            )
        end
    end
    pulse_history = zeros(
        n_controls,
        (
            n_inputs
            + (
                isnothing(boundary_values)
                ? 0
                : boundary_padding[1] + boundary_padding[2]
            )
        )
        * sampling_rate,
    )
    if !isnothing(boundary_values)
        pulse_history[:, 1 : boundary_padding[1] * sampling_rate] .= (
            boundary_values[:, 1]
        )
        pulse_history[
            :, end - boundary_padding[2] * sampling_rate + 1 : end
        ] .= boundary_values[:, 2]
    end
    return FilterShaping(
        pulse_history,
        zeros(
            n_controls,
            (n_inputs + !isnothing(boundary_values) * boundary_padding[2])
            * sampling_rate,
        ),
        sampling_rate,
        boundary_values,
        boundary_padding,
        kernel,
    )
end

function (s::FilterShaping{Nothing})(
    t_step::Int, epsilon_t::AbstractVector{Float64}
)
    t_sub = range(s.sampling_rate * (t_step - 1) + 1, s.sampling_rate * t_step)
    s.pulse_history[:, t_sub] .= epsilon_t
    for i in t_sub
        for n in 1 : t_step * s.sampling_rate
            s.shaped_pulse_history[:, i] += (
                s.pulse_history[:, n] * s.kernel((i - n) / s.sampling_rate)
            )
        end
    end
    return t_sub, s.shaped_pulse_history[:, t_sub]
end

function (s::FilterShaping{Matrix{Float64}})(
    t_step::Int, epsilon_t::AbstractVector{Float64}
)
    if (
        t_step * s.sampling_rate
        < (
            size(s.pulse_history, 2)
            - (s.boundary_padding[1] + s.boundary_padding[2]) * s.sampling_rate
        )
    )
        t_sub = range(
            s.sampling_rate * (t_step + s.boundary_padding[1] - 1) + 1,
            s.sampling_rate * (t_step + s.boundary_padding[1]),
        )
        s.pulse_history[:, t_sub] .= epsilon_t
        for i in t_sub
            t = i - s.boundary_padding[1] * s.sampling_rate
            for n in 1 : (t_step + s.boundary_padding[1]) * s.sampling_rate
                s.shaped_pulse_history[:, t] += (
                    s.pulse_history[:, n] * s.kernel((i - n) / s.sampling_rate)
                )
            end
        end
        return (
            t_sub .- s.boundary_padding[1] * s.sampling_rate,
            s.shaped_pulse_history[
                :, t_sub .- s.boundary_padding[1] * s.sampling_rate
            ],
        )
    end
    t_sub = range(
        s.sampling_rate * (t_step + s.boundary_padding[1] - 1) + 1,
        size(s.pulse_history, 2),
    )
    s.pulse_history[:, t_sub[1:s.sampling_rate]] .= epsilon_t
    for i in t_sub
        t = i - s.boundary_padding[1] * s.sampling_rate
        for n in range(
            1,
            (t_step + (s.boundary_padding[1] + s.boundary_padding[2]))
            * s.sampling_rate,
        )
            s.shaped_pulse_history[:, t] += (
                s.pulse_history[:, n] * s.kernel((i - n) / s.sampling_rate)
            )
        end
    end
    return (
        t_sub .- s.boundary_padding[1] * s.sampling_rate,
        s.shaped_pulse_history[
            :, t_sub .- s.boundary_padding[1] * s.sampling_rate
        ],
    )
end

function reset!(s::FilterShaping{Nothing})
    s.pulse_history .= zero(Float64)
    s.shaped_pulse_history .= zero(Float64)
    return nothing
end

function reset!(s::FilterShaping{Matrix{Float64}})
    s.pulse_history[
        :,
        s.boundary_padding[1] * s.sampling_rate +
        1 : end -
        s.boundary_padding[2] * s.sampling_rate
    ] .= zero(Float64)
    s.shaped_pulse_history .= zero(Float64)
    return nothing
end

_n_inpts(s::FilterShaping{Nothing}) = size(s.pulse_history, 2) ÷ s.sampling_rate

function _n_inpts(s::FilterShaping{Matrix{Float64}})
    return (
        (
            size(s.pulse_history, 2)
            - s.sampling_rate * (s.boundary_padding[1] + s.boundary_padding[2])
        )
        ÷ s.sampling_rate
    )
end

_n_ts(s::FilterShaping) = size(s.shaped_pulse_history, 2)
