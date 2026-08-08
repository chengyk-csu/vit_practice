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

def B32Config():
    config = ml_collections.ConfigDict()
    config.image_size = (224,224)
    config.patches = ml_collections.ConfigDict({"size":(32,32)})
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

def L16Config():
    config = ml_collections.ConfigDict()
    config.image_size = (224,224)
    config.patches = ml_collections.ConfigDict({"size":(16,16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.attention_heads = 16
    config.transformer.head_dim = 64
    config.transformer.num_layers = 24
    config.transformer.attn_dropout = 0.0
    config.transformer.dropout = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config

def L32Config():
    config = ml_collections.ConfigDict()
    config.image_size = (224,224)
    config.patches = ml_collections.ConfigDict({"size":(32,32)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.attention_heads = 16
    config.transformer.head_dim = 64
    config.transformer.num_layers = 24
    config.transformer.attn_dropout = 0.0
    config.transformer.dropout = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config

def H14Config():
    config = ml_collections.ConfigDict()
    config.image_size = (224,224)
    config.patches = ml_collections.ConfigDict({"size":(14,14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.attention_heads = 16
    config.transformer.head_dim = 80
    config.transformer.num_layers = 32
    config.transformer.attn_dropout = 0.0
    config.transformer.dropout = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config

def get_testing_Config():
    config = ml_collections.ConfigDict()
    config.image_size = (224,224)
    config.patches = ml_collections.ConfigDict({"size":(16,16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.attention_heads = 1
    config.transformer.head_dim = 1
    config.transformer.num_layers = 1
    config.transformer.attn_dropout = 0.0
    config.transformer.dropout = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config

def get_r50_b16_Config():
    config = ml_collections.ConfigDict()
    config.image_size = (224,224)
    config.patches = ml_collections.ConfigDict({"grid":(14,14)})
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
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3,4,9)
    config.resnet.width_factor = 1
    return config
