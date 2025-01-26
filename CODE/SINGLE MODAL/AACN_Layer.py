import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class AACN_Layer(nn.Module):
    def __init__(self, in_channels, k=0.25, v=0.25, kernel_size=3, num_heads=8, image_size=224, inference=False):
        super(AACN_Layer, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dk = math.floor((in_channels*k)/num_heads)*num_heads 
        # Paper: A minimum of 20 dimensions per head for the keys
        if self.dk / num_heads < 20:
            self.dk = num_heads * 20
        self.dv = math.floor((in_channels*v)/num_heads)*num_heads
        
        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dv % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"  
        
        self.padding = (self.kernel_size - 1) // 2
        
        self.conv_out = nn.Conv2d(self.in_channels, self.in_channels - self.dv, self.kernel_size, padding=self.padding).to(device)
        self.kqv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = nn.Conv2d(self.dv, self.dv, 1).to(device)
        
        # Positional encodings
        self.rel_encoding_h = nn.Parameter(torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        self.rel_encoding_w = nn.Parameter(torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        
        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)
         
    def forward(self, x):
        batch_size, _, height, width = x.size()
        dkh = self.dk // self.num_heads
        dvh = self.dv // self.num_heads
        flatten_hw = lambda x, depth: torch.reshape(x, (batch_size, self.num_heads, height * width, depth))

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        k, q, v = torch.split(kqv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (dkh ** -0.5)
        
        # After splitting, shape is [batch_size, num_heads, height, width, dkh or dvh]
        k = self.split_heads_2d(k, self.num_heads)
        q = self.split_heads_2d(q, self.num_heads)
        v = self.split_heads_2d(v, self.num_heads)
        
        # [batch_size, num_heads, height*width, height*width]
        qk = torch.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh).transpose(2, 3))

        qr_h, qr_w = self.relative_logits(q)
        qk += qr_h
        qk += qr_w

        weights = F.softmax(qk, dim=-1)
        
        if self.inference:
            self.weights = nn.Parameter(weights)
            
        attn_out = torch.matmul(weights, flatten_hw(v, dvh))
        attn_out = torch.reshape(attn_out, (batch_size, self.num_heads, self.dv // self.num_heads, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        # Project heads
        attn_out = self.attn_out(attn_out)
        return torch.cat((self.conv_out(x), attn_out), dim=1)

    # Split channels into multiple heads.
    def split_heads_2d(self, inputs, num_heads):
        batch_size, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads, height, width, depth // num_heads)
        split_inputs = torch.reshape(inputs, ret_shape)
        return split_inputs
    
    # Combine heads (inverse of split heads 2d).
    def combine_heads_2d(self, inputs):
        batch_size, num_heads, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads * depth, height, width)
        return torch.reshape(inputs, ret_shape)
    
    # Compute relative logits for both dimensions.
    def relative_logits(self, q):
        _, num_heads, height, width, dkh = q.size()
        rel_logits_w = self.relative_logits_1d(q, self.rel_encoding_w, height, width, num_heads,  [0, 1, 2, 4, 3, 5])
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.rel_encoding_h, width, height, num_heads,  [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w
    
    # Compute relative logits along one dimenion.
    def relative_logits_1d(self, q, rel_k, height, width, num_heads, transpose_mask):
        rel_logits = torch.einsum('bhxyd,md->bxym', q, rel_k)
        # Collapse height and heads
        rel_logits = torch.reshape(rel_logits, (-1, height, width, 2 * width - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it
        rel_logits = torch.reshape(rel_logits, (-1, height, width, width))
        # Tile for each head
        rel_logits = torch.unsqueeze(rel_logits, dim=1)
        rel_logits = rel_logits.repeat((1, num_heads, 1, 1, 1))
        # Tile height / width times
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, height, 1, 1))
        # Reshape for adding to the logits.
        rel_logits = rel_logits.permute(transpose_mask)
        rel_logits = torch.reshape(rel_logits, (-1, num_heads, height * width, height * width))
        return rel_logits

    # Converts tensor from relative to absolute indexing.
    def rel_to_abs(self, x):
        # [batch_size, num_heads*height, L, 2L−1]
        batch_size, num_heads, L, _ = x.size()
        # Pad to shift from relative to absolute indexing.
        col_pad = torch.zeros((batch_size, num_heads, L, 1)).to(device)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (batch_size, num_heads, L * 2 * L))
        flat_pad = torch.zeros((batch_size, num_heads, L - 1)).to(device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        # Reshape and slice out the padded elements.
        final_x = torch.reshape(flat_x_padded, (batch_size, num_heads, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

class AACN_EfficientNet_Layer(nn.Module):
    def __init__(self, in_channels, k, v, kernel_size, num_heads, image_size, inference):
        super(AACN_Layer, self).__init__()
        self.num_heads = num_heads
        self.dk = int(k * in_channels)
        self.dv = int(v * in_channels)
        self.inference = inference
        self.height, self.width = image_size, image_size

        self.kqv_conv = nn.Conv2d(in_channels, self.dk + self.dk + self.dv, kernel_size=kernel_size, padding=kernel_size // 2)
        self.output_conv = nn.Conv2d(self.dv, in_channels, kernel_size=1)

    def split_heads_2d(self, x, num_heads):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, num_heads, channels // num_heads, height, width)
        return x

    def relative_logits(self, q):
        batch_size, num_heads, height, width, dkh = q.size()
        relative_logits_h = torch.einsum('bnhwc,bnhwc->bnhw', q, q)
        relative_logits_w = torch.einsum('bnhwc,bnhwc->bnhw', q, q)
        return relative_logits_h, relative_logits_w

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.size()
        kqv = self.kqv_conv(x)

        k, q, v = torch.split(kqv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (self.dk ** -0.5)

        q = self.split_heads_2d(q, self.num_heads)
        k = self.split_heads_2d(k, self.num_heads)
        v = self.split_heads_2d(v, self.num_heads)

        flatten_hw = lambda x, depth: torch.reshape(x, (batch_size, self.num_heads, height * width, depth))
        qk = torch.matmul(flatten_hw(q, self.dk), flatten_hw(k, self.dk).transpose(2, 3))
        qr_h, qr_w = self.relative_logits(q)
        qk += qr_h + qr_w

        attn = F.softmax(qk, dim=-1)
        attn = torch.matmul(attn, flatten_hw(v, self.dv))

        attn = attn.view(batch_size, self.num_heads, height, width, self.dv // self.num_heads)
        attn = attn.permute(0, 2, 3, 1, 4).contiguous()
        attn = attn.view(batch_size, height, width, self.dv)
        attn = attn.permute(0, 3, 1, 2).contiguous()

        output = self.output_conv(attn)
        return output

class RelativeLogits(nn.Module):
    def __init__(self, height, width, num_heads, channels):
        super(RelativeLogits, self).__init__()
        self.rel_encoding_h = nn.Parameter(torch.randn(num_heads, channels, 2 * height - 1), requires_grad=True)
        self.rel_encoding_w = nn.Parameter(torch.randn(num_heads, channels, 2 * width - 1), requires_grad=True)

    def forward(self, q, height, width):
        rel_logits_h = self.relative_logits_1d(q, self.rel_encoding_h, height, width, transpose_mask=[0, 1, 2, 4, 3, 5])
        rel_logits_w = self.relative_logits_1d(torch.transpose(q, 3, 4), self.rel_encoding_w, width, height, transpose_mask=[0, 1, 2, 4, 3, 5])
        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, height, width, transpose_mask):
        # Debug prints to check dimensions
        # print(f"q shape: {q.shape}")
        # print(f"rel_k shape: {rel_k.shape}")

        # Adjust the einsum equation to match the dimensions
        rel_logits = torch.einsum('bhxyc,hmd->bhxym', q, rel_k)  # Corrected einsum equation

        # Debug prints to check dimensions
        # print(f"rel_logits shape before reshape: {rel_logits.shape}")

        return rel_logits

class AACN_VGG_Layer(nn.Module):
    def __init__(self, in_channels, k=0.25, v=0.25, kernel_size=3, num_heads=8, image_size=224, inference=False):
        super(AACN_VGG_Layer, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dk = math.floor((in_channels * k) / num_heads) * num_heads
        if self.dk / num_heads < 20:
            self.dk = num_heads * 20
        self.dv = math.floor((in_channels * v) / num_heads) * num_heads
        
        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dv % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"  
        
        self.padding = (self.kernel_size - 1) // 2
        
        # VGG-style layers (no residual connections, simple Conv2d layers)
        self.conv_out = nn.Conv2d(self.in_channels, self.in_channels - self.dv, self.kernel_size, padding=self.padding)
        self.kqv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = nn.Conv2d(self.dv, self.dv, 1)
        
        # Positional encodings
        self.rel_encoding_h = nn.Parameter(torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        self.rel_encoding_w = nn.Parameter(torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        
        # For accessing attention weights during inference
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)
         
    def forward(self, x):
        batch_size, _, height, width = x.size()
        dkh = self.dk // self.num_heads
        dvh = self.dv // self.num_heads
        flatten_hw = lambda x, depth: torch.reshape(x, (batch_size, self.num_heads, height * width, depth))

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        k, q, v = torch.split(kqv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (dkh ** -0.5)
        
        # After splitting, shape is [batch_size, num_heads, height, width, dkh or dvh]
        k = self.split_heads_2d(k, self.num_heads)
        q = self.split_heads_2d(q, self.num_heads)
        v = self.split_heads_2d(v, self.num_heads)
        
        # [batch_size, num_heads, height*width, height*width]
        qk = torch.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh).transpose(2, 3))

        qr_h, qr_w = self.relative_logits(q)
        qk += qr_h
        qk += qr_w

        weights = F.softmax(qk, dim=-1)
        
        if self.inference:
            self.weights = nn.Parameter(weights)
            
        attn_out = torch.matmul(weights, flatten_hw(v, dvh))
        attn_out = torch.reshape(attn_out, (batch_size, self.num_heads, self.dv // self.num_heads, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        # Project heads
        attn_out = self.attn_out(attn_out)
        return torch.cat((self.conv_out(x), attn_out), dim=1)

    def split_heads_2d(self, inputs, num_heads):
        batch_size, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads, height, width, depth // num_heads)
        split_inputs = torch.reshape(inputs, ret_shape)
        return split_inputs
    
    def combine_heads_2d(self, inputs):
        batch_size, num_heads, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads * depth, height, width)
        return torch.reshape(inputs, ret_shape)
    
    def relative_logits(self, q):
        _, num_heads, height, width, dkh = q.size()
        rel_logits_w = self.relative_logits_1d(q, self.rel_encoding_w, height, width, num_heads, [0, 1, 2, 4, 3, 5])
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.rel_encoding_h, width, height, num_heads, [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, height, width, num_heads, transpose_mask):
        rel_logits = torch.einsum('bhxyd,md->bxym', q, rel_k)

        # Correct calculation of expected elements per batch
        elements_per_batch = height * width * (2 * width - 1) * num_heads

        # Reshape directly using the known elements per batch
        rel_logits = rel_logits.view(-1, elements_per_batch)
        rel_logits = rel_logits.view(-1, height, width, 2 * width - 1, num_heads)
        rel_logits = self.rel_to_abs(rel_logits)

        return rel_logits

    def rel_to_abs(self, x):
        batch_size, num_heads, L, _ = x.size()
        col_pad = torch.zeros((batch_size, num_heads, L, 1)).to(x.device)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (batch_size, num_heads, L * 2 * L))
        flat_pad = torch.zeros((batch_size, num_heads, L - 1)).to(x.device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = torch.reshape(flat_x_padded, (batch_size, num_heads, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x