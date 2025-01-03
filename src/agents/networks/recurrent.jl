struct Recurrent{
    E <: Union{Embedding, Chain}, CLN <: CustomLayerNorm, R <: GRUv3Cell
}
    embedding_layer::E
    layer_norm_pre::CLN
    recurrent_layer::R
    layer_norm_post::CLN
end

@layer :noexpand Recurrent

"""
    Recurrent(
        observation_dim::Int, action_dim::Int, continuous::Bool; kwargs...
    )

Recurrent (GRU) model for partially observable quantum environments.

GRU model which allows including a memory of past actions and observations when
an agent has limited information of the model. For state control, measurement
outcomes are single outcomes of a operator, whereas for gate control,
measurement corresponds to a partial or full tomography of the system.

Args:
  * `observation_dim`: Dimension of the observation space.
  * `action_dim`: Dimension of the action space.
  * `continuous`: Whether the observation and space is discrete
        (typically for state control (single-shot experiments)) or continuous
        (corresponding to a partial / noisy tomography).

Kwargs:
  * `embedding_dim`: Dimension of the embedding space (default: `128`).
  * `ffn_activation`: Activation function for the feed-forward network (default:
        [`NNlib.relu`]()).
  * `ffn_bias`: Whether to include a bias in the feed-forward network (default:
        `true`).
  * `ln_bias`: Whether to include a learnable bias in the layer norm layer (
        default: `true`).
  * `recurrent_bias`: Whether to include a bias in the recurrent layer (default:
        `true`).
  * `init`: Initialisation function for the weights (default:
        [`NNlib.glorot_normal`]()).
  * `rng`: The random number generator to use (default: `Random.default_rng()`).

Fields:
  * `embedding_layer`: Embedding layer for the inputs.
  * `layer_norm_pre`: Layer norm before the recurrent layer.
  * `recurrent_layer`: GRU cell.
  * `layer_norm_post`: Layer norm after the recurrent layer.
"""
function Recurrent(
    observation_dim::Int,
    action_dim::Int,
    continuous::Bool;
    embedding_dim::Int = 128,
    ffn_activation::Function = relu,
    ffn_bias::Bool = true,
    ln_bias::Bool = true,
    recurrent_bias::Bool = true,
    init::Function = glorot_normal,
    rng::AbstractRNG = default_rng(),
)
    if continuous
        embedding_layer = Chain(
            Dense(
                observation_dim + action_dim,
                embedding_dim;
                bias=ffn_bias,
                init=init(rng),
            ),
            ffn_activation,
        )
    else
        embedding_layer = Embedding(
            observation_dim + action_dim, embedding_dim; init=init(rng)
        )
    end
    layer_norm_pre = CustomLayerNorm(embedding_dim; bias=ln_bias, eps=1f-5)
    recurrent_layer = GRUv3Cell(
        embedding_dim,
        embedding_dim;
        bias=recurrent_bias,
        init_kernel=init(rng),
        init_recurrent_kernel=init(rng),
    )
    layer_norm_post = CustomLayerNorm(embedding_dim; bias=ln_bias, eps=1f-5)
    return Recurrent(
        embedding_layer, layer_norm_pre, recurrent_layer, layer_norm_post
    )
end

function Base.show(io::IO, m::Recurrent)
    print(io, "Recurrent(")
    print(io, _input_dim(m), " => ", _embedding_dim(m))
    print(io, isa(m.embedding_layer, Embedding) ? ", false" : ", true")
    print(io, ")")
    return nothing
end

function _input_dim(m::Recurrent)
    if isa(m.embedding_layer, Embedding)
        return size(m.embedding_layer.weight, 2)
    end
    return size(m.embedding_layer[1].weight, 2)
end

function _embedding_dim(m::Recurrent)
    if isa(m.embedding_layer, Embedding)
        return size(m.embedding_layer.weight, 1)
    end
    return size(m.embedding_layer[1].weight, 1)
end
