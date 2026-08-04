import torch
import torch.nn as nn
from configs import B16Config
import math

class Embeddings(nn.Module):
    def __init__(self,config:B16Config):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=16,
            stride=16,
            padding=0
        )
        self.CLS_token = nn.Parameter(torch.zeros(1,1,config.hidden_size))
        self.num_patches = (config.image_size[0]//config.patches.size[0])**2
        self.position_embeddings = nn.Parameter(torch.zeros(1,self.num_patches+1,config.hidden_size))
        self.dropout = nn.Dropout(config.transformer.dropout)
    def forward(self,x):
        x = self.token_embeddings(x)
        x = torch.reshape(x,(-1,self.config.hidden_size,self.num_patches))
        x = torch.transpose(x,1,2)
        CLS_tokens = self.CLS_token.expand(x.shape[0],-1,-1)
        inputs = torch.concat([CLS_tokens,x],dim=1)
        inputs = inputs+self.position_embeddings
        inputs = self.dropout(inputs)
        return inputs

class Attention(nn.Module):
    def __init__(self,vis,config:B16Config):
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
    def __init__(self,config:B16Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size,config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim,config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.transformer.dropout)
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
    def __init__(self,vis,config:B16Config):
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

class Encoder(nn.Module):
    def __init__(self,vis,config:B16Config):
        super().__init__()
        self.config = config
        self.vis = vis
        self.layers = nn.ModuleList([
            Block(vis=self.vis,config=config)
            for _ in range(config.transformer.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size,eps=1e-6)
    def forward(self,x):
        attn_weights = []
        for layer in self.layers:
            x,weights = layer(x)
            if self.vis:
                attn_weights.append(weights)
        x = self.norm(x)
        return x,attn_weights

class Transformer(nn.Module):
    def __init__(self,vis,config:B16Config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(vis,config)
    def forward(self,x):
        inputs = self.embeddings(x)
        outputs,attn_weights = self.encoder(inputs)
        return outputs,attn_weights

class VisionTransformer(nn.Module):
    def __init__(self,num_classes,zero_head,vis,config:B16Config):
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
            loss = nn.CrossEntropyLoss(logits,labels)
            return loss
        else:
            return logits,attn_weights


