import torch
import torch.nn as nn
from . import configs
import math
from scipy import ndimage
import numpy as np
import torch.nn.functional as F
from .modeling_resnet import ResNetV2

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights,conv=False):
    weights = torch.from_numpy(weights)
    if conv:
        weights = weights.permute(3,2,0,1)
    return weights

def pjoin(*parts):
    key = "/".join(parts)
    return key

class Embeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        if config.patches.get("grid") is not None:
            self.hybrid = True
            patch_size = (config.image_size[0]//16//config.patches.grid[0],config.image_size[1]//16//config.patches.grid[1])
            n_patches = config.patches.grid[0]*config.patches.grid[1]
        else:
            self.hybrid = False
            patch_size = config.patches.size
            n_patches = (config.image_size[0]//patch_size[0])*(config.image_size[1]//patch_size[1])
        if self.hybrid:
            self.hybrid_model = ResNetV2(config.resnet.width_factor,config.resnet.num_layers)
            in_channels = self.hybrid_model.width*16
        else:
            in_channels = 3
        self.token_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.CLS_token = nn.Parameter(torch.zeros(1,1,config.hidden_size))
        self.num_patches = n_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1,self.num_patches+1,config.hidden_size))
        self.dropout = nn.Dropout(config.transformer.dropout)
    def forward(self,x):
        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.token_embeddings(x)
        x = x.flatten(2)
        x = torch.transpose(x,1,2)
        CLS_tokens = self.CLS_token.expand(x.shape[0],-1,-1)
        inputs = torch.concat([CLS_tokens,x],dim=1)
        inputs = inputs+self.position_embeddings
        inputs = self.dropout(inputs)
        return inputs

class Attention(nn.Module):
    def __init__(self,vis,config):
        super().__init__()
        self.config = config
        self.vis = vis
        self.num_heads = config.transformer.attention_heads
        self.head_dim = config.transformer.head_dim
        self.all_head_size = self.num_heads*self.head_dim
        self.q_proj = nn.Linear(config.hidden_size,self.all_head_size)
        self.k_proj = nn.Linear(config.hidden_size,self.all_head_size)
        self.v_proj = nn.Linear(config.hidden_size,self.all_head_size)
        self.o_proj = nn.Linear(self.all_head_size,config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attn_dropout)
        self.proj_dropout = nn.Dropout(config.transformer.attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
    def transpose_for_scores(self,qkv):
        qkv = qkv.reshape(qkv.shape[0],qkv.shape[1],self.num_heads,self.head_dim)
        qkv = qkv.transpose(1,2)
        return qkv
    def forward(self,inputs):
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        attention_scores = q @ k.transpose(-1,-2)
        attention_scores = attention_scores/math.sqrt(self.head_dim)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        outputs = attention_probs @ v
        outputs = outputs.transpose(1,2)
        outputs = outputs.reshape(outputs.shape[0],outputs.shape[1],-1)
        outputs = self.o_proj(outputs)
        outputs = self.proj_dropout(outputs)
        return outputs,weights

class Mlp(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size,config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim,config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.transformer.dropout)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias,std=1e-6)
        nn.init.normal_(self.fc2.bias,std=1e-6)
    def forward(self,inputs):
        inputs = self.fc1(inputs)
        inputs = self.gelu(inputs)
        inputs = self.dropout(inputs)
        inputs = self.fc2(inputs)
        inputs = self.dropout(inputs)
        return inputs

class Block(nn.Module):
    def __init__(self,vis,config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.norm1 = nn.LayerNorm(config.hidden_size,eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size,eps=1e-6)
        self.attention = Attention(vis=vis,config=config)
        self.mlp = Mlp(config)
    def forward(self,x):
        residual = x
        x = self.norm1(x)
        x, weights = self.attention(x)
        x = x+residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x+residual
        return x,weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,                                                                                  self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attention.q_proj.weight.copy_(query_weight)
            self.attention.k_proj.weight.copy_(key_weight)
            self.attention.v_proj.weight.copy_(value_weight)
            self.attention.o_proj.weight.copy_(out_weight)
            self.attention.q_proj.bias.copy_(query_bias)
            self.attention.k_proj.bias.copy_(key_bias)
            self.attention.v_proj.bias.copy_(value_bias)
            self.attention.o_proj.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.mlp.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            self.norm1.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.norm1.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.norm2.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.norm2.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self,vis,config):
        super().__init__()
        self.config = config
        self.vis = vis
        self.layer = nn.ModuleList([
            Block(vis=self.vis,config=config)
            for _ in range(config.transformer.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size,eps=1e-6)
    def forward(self,x):
        attn_weights = []
        for block in self.layer:
            x,weights = block(x)
            if self.vis:
                attn_weights.append(weights)
        x = self.norm(x)
        return x,attn_weights

class Transformer(nn.Module):
    def __init__(self,vis,config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(vis,config)
    def forward(self,x):
        inputs = self.embeddings(x)
        outputs,attn_weights = self.encoder(inputs)
        return outputs,attn_weights

class VisionTransformer(nn.Module):
    def __init__(self,num_classes,zero_head,vis,config):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(vis,config)
        self.classification_head = nn.Linear(config.hidden_size,num_classes)
    def forward(self,x,labels=None):
        outputs,attn_weights = self.transformer(x)
        CLS_TOKEN = outputs[:,0,:]
        logits = self.classification_head(CLS_TOKEN)
        if labels is not None:
            loss = F.cross_entropy(logits,labels)
            return loss
        else:
            return logits,attn_weights
    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.classification_head.weight)
                nn.init.zeros_(self.classification_head.bias)
            else:
                self.classification_head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.classification_head.bias.copy_(np2th(weights["head/bias"]))
            self.transformer.embeddings.token_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.token_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.CLS_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)
                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
                gs_old = int(math.sqrt(len(posemb_grid)))
                gs_new = int(math.sqrt(ntok_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1,-1,posemb_grid.shape[-1])
                posemb = np.concatenate([posemb_tok,posemb_grid],axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            for i,block in self.transformer.encoder.layer.named_children():
                block.load_from(weights,n_block=i)
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'VIT-B_16':configs.B16Config(),
    'VIT-B_32':configs.B32Config(),
    'VIT-L_16':configs.L16Config(),
    'VIT-L_32':configs.L32Config(),
    'VIT-H_14':configs.H14Config(),
    'VIT-testing':configs.get_testing_Config(),
    'VIT-r50_b16':configs.get_r50_b16_Config()
}

