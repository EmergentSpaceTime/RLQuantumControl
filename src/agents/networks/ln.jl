# Custom LayerNorm layer exactly based on the Flux implementation but with one
# small change to allow optional bias.

struct CustomLayerNorm{F, D, T, N}
    λ::F
    diag::D
    ϵ::T
    size::NTuple{N, Int}
    affine::Bool
end

function CustomLayerNorm(
    size::Tuple{Vararg{Int}},
    λ = identity;
    affine::Bool = true,
    bias::Bool = true,
    eps::Real = 1f-5,
    ϵ = nothing,
)
    ε = _greek_ascii_depwarn(ϵ => eps, :LayerNorm, "ϵ" => "eps")
    if affine
        diag = Scale(size..., λ; bias=bias)
    else
        if λ != identity
            diag = Base.Fix1(broadcast, λ)
        else
            diag = identity
        end
    end
    return CustomLayerNorm(λ, diag, ε, size, affine)
end

CustomLayerNorm(size::Integer...; kw...) = CustomLayerNorm(Int.(size); kw...)

function CustomLayerNorm(size_act...; kw...)
    return CustomLayerNorm(Int.(size_act[1 : end - 1]), size_act[end]; kw...)
end

@layer CustomLayerNorm

function (a::CustomLayerNorm)(x::AbstractArray)
    @ignore_derivatives if a.diag isa Scale
        for d in 1:ndims(a.diag.scale)
            _size_check(a, x, d => size(a.diag.scale, d))
        end
    end
    eps = convert(float(eltype(x)), a.ϵ)
    return a.diag(normalise(x; dims=1:length(a.size), eps))
end

hasaffine(a::CustomLayerNorm) = a.affine

function Base.show(io::IO, l::CustomLayerNorm)
    print(io, "CustomLayerNorm(", join(l.size, ", "))
    l.λ === identity || print(io, ", ", l.λ)
    hasaffine(l) || print(io, "; affine=false")
    l.diag isa Scale && l.diag.bias == false && print(io, "; bias=false")
    print(io, ")")
    return nothing
end
