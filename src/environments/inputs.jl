"""Abstract callable struct that creates a output pulse from a given action from
the controller. Custom inputs must define `control_min` and `control_max` fields
which is a vector with length matching the number of controls and a
[`action_space`]() method.

These callables have the argument signature:
```math
    \\mathscr{I}(\\epsilon_{t - 1}, a_{t})\\rightarrow\\epsilon_{t}
```
"""
abstract type InputFunction <: Function end

_n_ctrls(i::InputFunction) = length(i.control_min)


struct IdentityInput <: InputFunction
    control_min::Vector{Float64}
    control_max::Vector{Float64}
end

"""
    IdentityInput(
        n_controls::Int,
        control_min::Vector{<:Real} = fill(-Inf, n_controls),
        control_max::Vector{<:Real} = fill(Inf, n_controls),
    )

Identity callable generating an action space of
``[\\epsilon_{\\text{min}}``, ``\\epsilon_{\\text{max}}]``:
```math
    \\mathscr{I}(\\epsilon_{t - 1}, a_{t}) = a_{t}
```

Args:
  * `n_controls`: Number of control pulses.

Kwargs:
  * `control_min`: Minimum input values (default: [`fill(-Inf, n_controls)`]()).
  * `control_max`: Maximum input values (default: [`fill(Inf, n_controls)`]()).

Fields:
  * `control_min`: Minimum input values.
  * `control_max`: Maximum input values.
"""
function IdentityInput(
    n_controls::Int;
    control_min::Vector{<:Real} = fill(-Inf, n_controls),
    control_max::Vector{<:Real} = fill(Inf, n_controls),
)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    if (
        !=(length(control_min), n_controls)
        | !=(length(control_max), n_controls)
    )
        throw(
            DimensionMismatch(
                "Length of `control_min` and `control_max` must be equal to"
                * " `n_controls`."
            )
        )
    end
    return IdentityInput(control_min, control_max)
end

(::IdentityInput)(::AbstractVector{Float64}, a::Vector{Float64}) = a

action_space(i::IdentityInput) = ClosedInterval.(i.control_min, i.control_max)


struct StepInput <: InputFunction
    control_min::Vector{Float64}
    control_max::Vector{Float64}
    delta_control::Vector{Float64}
end

"""
    StepInput(
        n_controls::Int;
        control_min::Vector{<:Real} = fill(-Inf, n_controls),
        control_max::Vector{<:Real} = fill(Inf, n_controls),
        delta_control::Vector{<:Real} = fill(Inf, n_controls),
    )

Step-limited input callable that generates an action space of
``[-\\Delta\\epsilon``, ``\\Delta\\epsilon]`` such that:
```math
    \\mathscr{I}(\\epsilon_{t - 1}, a_{t}) = \\epsilon_{t - 1} + a_{t}
```

Args:
  * `n_controls`: Number of control pulses.

Kwargs:
  * `control_min`: Minimum input values (default: [`fill(-Inf, n_controls)`]()).
  * `control_max`: Maximum input values (default: [`fill(Inf, n_controls)`]()).
  * `delta_control`: Maximal change in input values (default:
        [`fill(Inf, n_controls)`]()).

Fields:
  * `control_min`: Minimum input values.
  * `control_max`: Maximum input values.
  * `delta_control`: Maximal change in input values.
"""
function StepInput(
    n_controls::Int;
    control_min::Vector{<:Real} = fill(-Inf, n_controls),
    control_max::Vector{<:Real} = fill(Inf, n_controls),
    delta_control::Vector{<:Real} = fill(Inf, n_controls),
)
    n_controls <= 0 && throw(ArgumentError("`n_controls` must be >= 1."))
    if (
        !=(length(control_min), n_controls)
        | !=(length(control_max), n_controls)
        | !=(length(delta_control), n_controls)
    )
        throw(
            DimensionMismatch(
                "Length of `control_min`, `control_max`, and `delta_control`"
                * "must be equal to `n_controls`."
            )
        )
    end
    return StepInput(control_min, control_max, delta_control)
end

function (i::StepInput)(
    control_tm1::AbstractVector{Float64}, a::Vector{Float64}
)
    return clamp(control_tm1 + a, i.control_min, i.control_max)
end

action_space(i::StepInput) = ClosedInterval.(-i.delta_control, i.delta_control)


"""
    is_valid_input(action_space::ClosedInterval{Float64}, a::Vector{Float64})

Check if the given input is within the action space.

Args:
  * `action_space`: The action space.
  * `a`: Action to check.

Returns:
  * `Bool`: Whether the action is within the action space.
"""
function is_valid_input(
    action_space::Vector{ClosedInterval{Float64}}, a::Vector{Float64}
)
    for i in eachindex(a)
        in(a[i], action_space[i]) || return false
    end
    return true
end
