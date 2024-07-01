struct TransformerBlock{
    CLN <: CustomLayerNorm,
    MHA <: MultiHeadAttention,
    D <: Dropout,
    FFN <: Chain,
}
    layer_norm_pre::CLN
    multi_head_attention::MHA
    residual_dropout::D
    layer_norm_post::CLN
    feed_forward_network::FFN
end

@layer :expand TransformerBlock

function TransformerBlock(
    ;
    embedding_dim::Int = 128,
    n_heads::Int = 4,
    attention_bias::Bool = true,
    attention_dropout::Float32 = 0.0f0,
    residual_dropout::Float32 = 0.0f0,
    intermediate_dim::Int = 4 * embedding_dim,
    ffn_activation::Function = gelu,
    ffn_bias::Bool = true,
    ln_bias::Bool = true,
    init::Function = glorot_normal,
)
    return TransformerBlock(
        CustomLayerNorm(embedding_dim; bias=ln_bias),
        MultiHeadAttention(
            embedding_dim;
            nheads=n_heads,
            bias=attention_bias,
            init=init,
            dropout_prob=attention_dropout,
        ),
        Dropout(residual_dropout),
        CustomLayerNorm(embedding_dim; bias=ln_bias),
        Chain(
            Dense(embedding_dim, intermediate_dim; init=init, bias=ffn_bias),
            ffn_activation,
            Dense(intermediate_dim, embedding_dim; init=init, bias=ffn_bias),
            Dropout(residual_dropout),
        ),
    )
end

function (m::TransformerBlock)(qkv::AbstractArray{Float32, 3})
    a, _ = m.multi_head_attention(
        m.layer_norm_pre(qkv); mask=make_causal_mask(qkv)
    )
    a = m.residual_dropout(a)
    a += qkv
    a += m.feed_forward_network(m.layer_norm_post(a))
    return a
end


struct GPT{
    TE <: Union{Embedding, Chain},
    PE <: Embedding,
    D <: Dropout,
    TB <: Chain{<:Tuple{Vararg{<:TransformerBlock}}},
    CLN <: CustomLayerNorm,
}
    token_embedding::TE
    position_embedding::PE
    embedding_dropout::D
    transformer_blocks::TB
    final_layer_norm::CLN
end

@layer GPT

"""
    GPT(
        observation_dim::Int,
        continuous_observation::Bool,
        sequence_length::Int;
        kwargs...
    )

GPT model for partially observable quantum environments.

Transformer model based on GPT-2 [radford2019language](@cite) which allows
including a memory of past actions and observations when an agent has limited
information of the model. For state control, measurement outcomes are single
outcomes of a operator, whereas for gate control, measurement corresponds to a
partial or full tomography of the system.

Args:
  * `observation_dim`: Dimension of the observation space.
  * `continuous_observation`: Whether the observation space is discrete
        (typically single-shot experiments) or continuous (corresponding to a
        partial / noisy tomography).
  * `sequence_length`: Length of the sequence of observations (number of
        discrete time steps in protocol).

Kwargs:
  * `embedding_dim`: Dimension of the embedding space (default: `128`).
  * `embedding_dropout`: Dropout probability for the embedding (default:
        `0.0f0`).
  * `n_blocks`: Number of transformer blocks (default: `2`).
  * `n_heads`: Number of attention heads (default: `8`).
  * `attention_bias`: Whether to include a bias in the attention layer (default:
        `false`).
  * `attention_dropout`: Dropout probability for the attention layer (default:
        `0.0f0`).
  * `residual_dropout`: Dropout probability for the residual connections
        (default: `0.0f0`).
  * `intermediate_dim`: Dimension of the intermediate layer in the feed-forward
        network (default: `4 × embedding_dim`).
  * `ffn_activation`: Activation function for the feed-forward network (default:
        [`NNlib.gelu`]()).
  * `ffn_bias`: Whether to include a bias in the feed-forward network (default:
        `true`).
  * `ln_bias`: Whether to include a learnable bias in the layer norm layer (
        default: `true`).
  * `init`: Initialisation function for the weights (default:
        [`NNlib.glorot_normal`]()).

Fields:
  * `token_embedding`: Embedding layer for the tokens.
  * `position_embedding`: Embedding layer for the positions.
  * `embedding_dropout`: Dropout layer for the embeddings.
  * `transformer_blocks`: Transformer blocks.
  * `final_layer_norm`: Layer norm for the final output.
"""
function GPT(
    observation_dim::Int,
    continuous_observation::Bool,
    sequence_length::Int;
    embedding_dim::Int = 128,
    embedding_bias::Bool = true,
    embedding_activation::Function = gelu,
    embedding_dropout::Float32 = 0.0f0,
    n_blocks::Int = 2,
    n_heads::Int = 4,
    attention_bias::Bool = true,
    attention_dropout::Float32 = 0.0f0,
    residual_dropout::Float32 = 0.0f0,
    intermediate_dim::Int = 4 * embedding_dim,
    ffn_activation::Function = gelu,
    ffn_bias::Bool = true,
    ln_bias::Bool = true,
    init::Function = glorot_normal,
)
    if continuous_observation
        token_embedding = Chain(
            Dense(
                observation_dim, embedding_dim; init=init, bias=embedding_bias
            ),
            embedding_activation,
        )
    else
        token_embedding = Embedding(observation_dim, embedding_dim; init=init)
    end
    transformer_blocks = Chain(
        [
            TransformerBlock(
                ;
                embedding_dim=embedding_dim,
                n_heads=n_heads,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
                intermediate_dim=intermediate_dim,
                ffn_activation=ffn_activation,
                ffn_bias=ffn_bias,
                ln_bias=ln_bias,
                init=init,
            )
            for _ in 1:n_blocks
        ]...
    )
    for block in transformer_blocks  # GPT-2 scaling
        block.multi_head_attention.out_proj.weight .= (
            init(embedding_dim, embedding_dim) / sqrt(2 * n_blocks)
        )
    end
    return GPT(
        token_embedding,
        Embedding(sequence_length, embedding_dim; init=init),
        Dropout(embedding_dropout),
        transformer_blocks,
        CustomLayerNorm(embedding_dim; bias=ln_bias),
    )
end

"""
    (m::GPT)(o::Union{AbstractArray{Float32, 3}, AbstractMatrix{Int}})

Forward pass of the GPT model.

Args:
  * `o`: Observations up to time step (or max number of time steps) with shape
        (`1:t`, `batch_size`) for discrete observations or
        (`observation_dim`, `1:t`, `batch_size`) for continuous observations.

Returns:
  * `AbstractArray{Float32, 3}`: Output of the transformer of shape
        (`embedding_dim`, `1:t`, `batch_size`).
"""
function (m::GPT)(o::Union{AbstractArray{Float32, 3}, AbstractMatrix{Int}})
    x = m.embedding_dropout(
        m.token_embedding(o) .+ m.position_embedding(1:size(o, ndims(o) - 1))
    )
    for block in m.transformer_blocks
        x = block(x)
    end
    return m.final_layer_norm(x)
end

function Base.show(io::IO, m::GPT)
    print(io, "GPT(")
    print(io, _input_dim(m), " => ", _embedding_dim(m))
    print(io, isa(m.token_embedding, Embedding) ? ", false" : ", true")
    print(io, ", ", size(m.position_embedding.weight, 2))
    print(io, ")")
    return nothing
end

function _input_dim(m::GPT)
    if isa(m.token_embedding, Embedding)
        return size(m.token_embedding.weight, 2)
    end
    return size(m.token_embedding[1].weight, 2)
end

function _embedding_dim(m::GPT)
    if isa(m.token_embedding, Embedding)
        return size(m.token_embedding.weight, 1)
    end
    return size(m.token_embedding[1].weight, 1)
end
