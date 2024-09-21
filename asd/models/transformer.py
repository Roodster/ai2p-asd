import copy
import math

import torch as th
import torch.nn as nn

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

from asd.models.base import BaseModel

class Attention(BaseModel):
    def __init__(self, args):
        super(Attention, self).__init__(args=args)
        
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(self.hidden_size, self.all_head_size).to(self.device)
        self.key = Linear(self.hidden_size, self.all_head_size).to(self.device)
        self.value = Linear(self.hidden_size, self.all_head_size).to(self.device)

        self.out = Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.attn_dropout = Dropout(p=0.2)
        self.proj_dropout = Dropout(p=0.2)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # print(f"Attention {hidden_states.shape}")

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = th.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = th.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        # print(f"Attention output {attention_output.shape}")

        return attention_output


class MLP(BaseModel):
    def __init__(self, args):
        super(MLP, self).__init__(args=args)
        self.fc1 = Linear(256, 256).to(self.device)
        self.fc2 = Linear(256, 256).to(self.device)
        self.act_fn = th.nn.functional.gelu
        self.dropout = Dropout(p=0.2)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # print(f"MLP {x.shape}")
        x = x.to(self.device)
        x = self.fc1(x)
        # print(f"MLP fc1 {x.shape}")

        x = self.act_fn(x)
        
        # print(f"MLP act fn {x.shape}")

        x = self.dropout(x)
        x = self.fc2(x)
        # print(f"MLP fc2 {x.shape}")

        x = self.dropout(x)
        return x


class Embeddings(BaseModel):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, args, img_size=(4, 256), patch_sizes=(16, 16), in_channels=1):
        super(Embeddings, self).__init__(args=args)
        img_size = _pair(img_size)

        patch_size = _pair((patch_sizes[0], patch_sizes[1]))
        n_patches = patch_size[0] * patch_size[1]

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=n_patches,
                                       kernel_size=patch_size,
                                       stride=patch_size).to(self.device)
        self.position_embeddings = nn.Parameter(th.zeros(1, n_patches + 1, 256)).to(self.device)
        self.cls_token = nn.Parameter(th.zeros(1, 1, 256)).to(self.device)

        self.dropout = Dropout(0.2)

    def forward(self, x):
        # print(f"Embeddings input {x.shape}")
        x = x.to(self.device)
        x = x.unsqueeze(1)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # print(f"Embeddings patch {x.shape}")

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = th.cat((cls_tokens, x), dim=1)
        # print(f"Embeddings after patch {x.shape}")
        # print(f"Embeddings position embedding {self.position_embeddings.shape}")

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        # print(f"Embeddings return {x.shape}")

        return embeddings


class Block(BaseModel):
    def __init__(self, args):
        super(Block, self).__init__(args=args)
        self.hidden_size = 256
        self.attention_norm = LayerNorm(256, eps=1e-6).to(self.device)
        self.ffn_norm = LayerNorm(256, eps=1e-6).to(self.device)
        self.ffn = MLP(args).to(self.device)
        self.attn = Attention(args).to(self.device)

    def forward(self, x):
        # print(f"Block {x.shape}")
        x = x.to(self.device)

        h = x
        # print(f"Block attention {x.shape}")

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        # print(f"Block ffn_norm {x.shape}")
        x = self.ffn_norm(x)
        # print(f"Block ffn {x.shape}")
        x = self.ffn(x)
        x = x + h
        # print(f"Block ffn {x.shape}")
        return x



class Encoder(BaseModel):
    def __init__(self, args, n_encoders=1):
        super(Encoder, self).__init__(args=args)
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(256, eps=1e-6).to(self.device)
        for _ in range(n_encoders):
            layer = Block(args)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        # print(f"Encoder {hidden_states.shape}")
        hidden_states = hidden_states.to(self.device)

        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        # print(f"Encoder encoded {encoded.shape}")

        return encoded


class Transformer(BaseModel):
    def __init__(self, args, img_size=(4)):
        super(Transformer, self).__init__(args=args)
        self.embeddings = Embeddings(args, img_size=img_size).to(self.device)
        self.encoder = Encoder(args).to(self.device)

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)

        # print(f"Transformer {input_ids.shape}")

        embedding_output = self.embeddings(input_ids)
        # print(f"Encoder encoded {embedding_output.shape}")

        encoded = self.encoder(embedding_output)
        # print(f"Encoder encoded {encoded.shape}")

        return encoded


class VisionTransformer(BaseModel):
    def __init__(self, args, img_size=256, num_classes=2, zero_head=False):
        super(VisionTransformer, self).__init__(args=args)
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.transformer = Transformer(args, img_size=img_size).to(self.device)
        self.head = Linear(256, num_classes).to(self.device)

    def forward(self, x, labels=None):
        x = x.to(self.device)
        # print(f"VisionTransformer {x.shape}")
        x = self.transformer(x)
        # print(f"VisionTransformer transformer {x.shape}")
        logits = self.head(x[:, 0])
        # print(f"VisionTransformer logits {logits.shape}")

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits



class GaussianNoise(BaseModel):
    
    def __init__(self, args):
        super().__init__(args=args)    
        self.mu = 0 
        self.std = 1
    
    def forward(self, x):
        return x + th.randn((1, 256, 4))
   

class CutRearrange(BaseModel):
    
    def __init__(self, args, n_segments=8):
        super().__init__(args=args)
        self.n_segments = n_segments
        
    def forward(self, x):
        x = x.to(self.device)

        # Assuming x has shape (batch_size, 256) or (256,)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if not present
        
        batch_size, signal_length = x.shape
        segment_length = signal_length // self.n_segments

        # Split the signal into segments
        x = x.view(batch_size, self.n_segments, segment_length)
        
        # Randomly permute the segments
        perm = th.randperm(self.n_segments)
        x = x[:, perm, :]
        
        # Merge the segments back into a single signal
        x = x.view(batch_size, -1)
        
        return x

class SSLTransformer(BaseModel):
    
    def __init__(self, args):
        super().__init__(args=args)    
        self.gaussian_noise = GaussianNoise()
        self.cut_rearrange = CutRearrange()
        
        self.encoder1 = VisionTransformer(args=args)
        self.encoder2 = VisionTransformer(args=args)
        
    def forward(self, x):

        x1 = self.gaussian_noise(x)
        x2 = self.cut_rearrange(x)
        
        p1 = self.encoder1(x1)
        p2 = self.encoder2(x2) 
        
        return p1, p2