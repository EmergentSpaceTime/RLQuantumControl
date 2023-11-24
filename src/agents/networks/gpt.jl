# """Contains GPT model for autoregressive sequence modeling, ideally for
# partially observable environments.
# """

struct TransformerBlock{
    LN <: LayerNorm, MHA <: MultiHeadAttention, D <: Dropout, FFN <: Chain
}
    layer_norm_q::LN
    layer_norm_kv::LN
    multi_head_attention::MHA
    residual_dropout::D
    layer_norm_post::LN
    feed_forward_network::FFN
end

@functor TransformerBlock

function TransformerBlock(
    embedding_dim::Int;
    n_heads::Int = 8,
    intermediate_dim::Int = 4 * embedding_dim,
    init::Function = glorot_uniform,
    attention_bias::Bool = false,
    attention_dropout::Float32 = 0.0f0,
    residual_dropout::Float32 = 0.0f0,
    ffn_activation::Function = gelu,
    ffn_bias::Bool = true,
)
    return TransformerBlock(
        LayerNorm(embedding_dim),
        LayerNorm(embedding_dim),
        MultiHeadAttention(
            embedding_dim;
            nheads=n_heads,
            bias=attention_bias,
            init=init,
            dropout_prob=attention_dropout,
        ),
        Dropout(residual_dropout),
        LayerNorm(embedding_dim),
        Chain(
            Dense(embedding_dim, intermediate_dim; init=init, bias=ffn_bias),
            ffn_activation,
            Dense(intermediate_dim, embedding_dim; init=init, bias=ffn_bias),
            Dropout(residual_dropout),
        ),
    )
end

function (m::TransformerBlock)(
    q::A, kv::A; mask::AbstractMatrix{Bool} = make_causal_mask(q)
) where {A <: AbstractArray{Float32, 3}}
    a, attention_scores = m.multi_head_attention(
        m.layer_norm_q(q), m.layer_norm_kv(kv); mask=mask
    )
    a .= m.residual_dropout(a)
    a .+= q
    a .+= m.feed_forward_network(m.layer_norm_post(a))
    return a, attention_scores
end


# """GPT model for partially observable quantum environments.

# Transformer model based on GPT-2 [1] which allows including a memory of past
# actions and observations when an agent has limited information of the model. For
# state control, measurement outcomes are single outcomes of a operator, whereas
# for gate control, "measurement" corresponds to a partial or full tomography of
# the system.

# Args:
#   * observation_dim: Dimension of the observation space.
#   * continuous_observation: Whether the observation space is discrete (typically
#         for state control) or continuous (corresponding to a partial
#         tomography).

# Kwargs:
#   * embedding_dim: Dimension of the embedding space (default: 128).
#   * n_heads: Number of attention heads (default: 8).
#   * n_blocks: Number of transformer blocks (default: 2).
#     intermediate_dim: Dimension of the intermediate layer in the feed-forward
#         network (default: 4 * [embedding_dim]@ref).
#   * init: Initialisation function for the weights (default: [`glorot_uniform`](@ref)).

# Fields:

# [1] Radford, A. et al. *Language models are unsupervised multitask learners*.
# [OpenAI blog](
#     https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf
# ) (2019).
# """
struct GPT{
    D <: Dense,
    E <: Embedding,
    DR <: Dropout,
    TB <: Chain{<:Tuple{Vararg{<:TransformerBlock}}},
    LN <: LayerNorm
}
    token_embedding::D
    positional_embedding::E
    embedding_dropout::DR
    transformer_blocks::TB
    final_layer_norm::LN
end

@functor GPT

function GPT(
    observation_dim::Int,
    continuous_observation::Bool,
    embedding_dim::Int;
    n_heads::Int = 8,
    intermediate_dim::Int = 4 * embedding_dim,
    init::Function = glorot_uniform,
    attention_bias::Bool = false,
    attention_dropout::Float32 = 0.0f0,
    residual_dropout::Float32 = 0.0f0,
    ffn_activation::Function = gelu,
    ffn_bias::Bool = true,
)
    return GPT(
        Dense(observation_dim, embedding_dim),
        Chain(
            [
                TransformerBlock(
                    embedding_dim,
                    n_heads;
                    dropout=dropout,
                    init=init,
                    activation=activation,
                    rng=rng,
                )
                for _ in 1:n_blocks
            ]...
        )
    )
end

function (m::GPT)(
    q::A, kv::A; mask::AbstractMatrix{Bool} = make_causal_mask(q)
) where {A <: AbstractArray{Float32, 3}}
    t = m.token_embedding(q)
    p = m.positional_embedding(kv)
end

# using Flux: glorot_normal
# LayerNorm(32; affine=true, bias=false)
# typeof(Chain(Dense(64, 256; init=glorot_normal), gelu, Dense(4, 5)))
# typeof(MultiHeadAttention(32; nheads=8, init=glorot_uniform, dropout_prob=0.1f0, bias=false))