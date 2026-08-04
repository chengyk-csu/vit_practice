import ml_collections
def B16Config():
    config = ml_collections.ConfigDict()
    config.image_size = (224,224)
    config.patches = ml_collections.ConfigDict({"size":(16,16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.attention_heads = 12
    config.transformer.head_dim = 64
    config.transformer.num_layers = 12
    config.transformer.attn_dropout = 0.0
    config.transformer.dropout = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config

