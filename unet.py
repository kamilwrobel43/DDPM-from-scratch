import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim, dropout = 0.2) -> None:
        super().__init__()

        self.group_norm1 = nn.GroupNorm(num_groups = 32, num_channels = in_channels)
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)

        self.t_proj = nn.Linear(in_features = t_dim, out_features = out_channels)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.group_norm2 = nn.GroupNorm(num_groups = 32, num_channels = out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels!=out_channels:
            self.fix_conv = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.fix_conv = nn.Identity()

    def forward(self, x, t):
        x_main = self.conv1(self.silu(self.group_norm1(x)))
        t = self.silu(self.t_proj(t)[:, :, None, None])
        x_main = x_main + t
        x_main = self.conv2(self.dropout(self.silu(self.group_norm2(x_main))))

        return x_main + self.fix_conv(x)

    


class Upsample(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.upsample_conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x):
        return self.upsample_conv(self.upsample(x))


class Downsample(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.downsample_conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 2, padding = 1)
    
    def forward(self, x):
        return self.downsample_conv(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, t_dim) -> None:
        super().__init__()

        self.t_dim = t_dim

        self.linear1 = nn.Linear(in_features = t_dim, out_features = t_dim * 4)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(in_features = t_dim*4, out_features = t_dim*4)

    def forward(self, x):
        pe = torch.zeros(x.shape[0], self.t_dim, device = x.device)
        position = x.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.t_dim, 2, device = x.device) * -(math.log(10000.0) // self.t_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        pe = self.silu(self.linear2(self.silu(self.linear1(pe))))

        return pe
        
        
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups = 32, num_channels = in_channels)
        self.qkv = nn.Linear(in_features = in_channels, out_features = 3*in_channels)
        self.proj = nn.Linear(in_features = in_channels, out_features = in_channels)

    
    def forward(self, x):
        B,C,H,W = x.shape

        identity = x

        x = self.group_norm(x)
        # (B, C, H, W) - > (B, H*W, C)
        x = x.reshape(B,C,H*W).permute(0,2,1)
        # (B, H*W, 3C) -> 3 x (B, H*W, C)
        q, k, v = self.qkv(x).chunk(3, dim = -1)

        # (B, H*W, C) -> (B, H*W, H*W)
        attention_scores = (q @ k.transpose(-2,-1)) / math.sqrt(C)

        attention_scores = attention_scores.softmax(dim = -1)

        # (B, H*W, H*W) -> (B, H*W, C)
        out = attention_scores @ v
        out = self.proj(out)
        # (B, H*W, C) -> (B, C, H, W)
        out = out.permute(0,2,1).view(B,C,H,W)
        out = out + identity

        return out

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, in_channels, n_heads) -> None:

        super().__init__()



        self.n_heads = n_heads

        self.group_norm = nn.GroupNorm(num_groups = 32, num_channels = in_channels)

        self.qkv = nn.Linear(in_features = in_channels, out_features = 3*in_channels)

        self.proj = nn.Linear(in_features = in_channels, out_features = in_channels)



    def forward(self, x):

        B,C,H,W = x.shape



        head_dim = C // self.n_heads



        identity = x

        x = self.group_norm(x)



        # (B, C, H, W) -> (B, H*W, C)

        x = x.reshape(B,C,H*W).permute(0,2,1)

        # (B, H*W, C) -> (B, H*W, 3C)

        qkv = self.qkv(x)

        # (B, H*W, 3C) -> (B, n_heads, H*W, 3*head_dim)

        qkv = qkv.reshape(B, H*W, self.n_heads,-1).permute(0, 2, 1, 3)

        # (B, n_heads, H*W, 3*head_dim) -> 3 x (B, n_heads, H*W, head_dim)

        q, k, v = qkv.chunk(3, dim = -1)

        div_term = math.sqrt(head_dim)



        # (B, n_heads, H*W, head_dim) -> (B, n_heads, H*W, H*W)

        attention_scores = (q @ k.transpose(-2,-1)) / div_term

        attention_scores = attention_scores.softmax(dim = -1)



        # (B, n_heads, H*W, H*W) -> (B, n_heads, H*W, head_dim)

        out = attention_scores @ v



        # (B, n_heads, H*W, head_dim) -> (B,H,W,C)

        out = out.permute(0,2,1,3).reshape(B,H,W,C)

        out = self.proj(out)

       

        # (B,H,W,C) -> (B,C,H,W)

        out = out.permute(0,3,1,2)

        out = out + identity

        return out
        

class UNet(nn.Module):
    def __init__(self, base_dim, in_channels = 3) -> None:
        super().__init__()
        
        t_dim = base_dim * 4
        self.sin_positional_encoding = SinusoidalPositionalEncoding(base_dim)
        self.initial_conv = nn.Conv2d(in_channels = in_channels, out_channels = base_dim, kernel_size = 3, stride = 1, padding = 1)

        self.downs = nn.ModuleList()
        self.downs.append(ResidualBlock(base_dim,base_dim,t_dim))
        self.downs.append(Downsample(base_dim))
        self.downs.append(ResidualBlock(base_dim,2*base_dim, t_dim))
        self.downs.append(Downsample(2*base_dim))
        self.downs.append(ResidualBlock(2*base_dim,4*base_dim, t_dim))
        self.downs.append(MultiHeadAttentionBlock(4*base_dim, n_heads = 4))
        self.downs.append(Downsample(4*base_dim))

        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResidualBlock(4*base_dim, 4*base_dim, t_dim))
        self.bottleneck.append(MultiHeadAttentionBlock(4*base_dim, n_heads = 4))
        self.bottleneck.append(ResidualBlock(4*base_dim, 4*base_dim, t_dim))

        self.ups = nn.ModuleList()
        self.ups.append(Upsample(4*base_dim))
        self.ups.append(ResidualBlock(2*4*base_dim, 4*base_dim, t_dim))
        self.ups.append(MultiHeadAttentionBlock(4*base_dim, n_heads = 4))
        self.ups.append(Upsample(4*base_dim))
        self.ups.append(ResidualBlock(4*base_dim+2*base_dim, 2*base_dim, t_dim))
        self.ups.append(Upsample(2*base_dim))
        self.ups.append(ResidualBlock(2*base_dim+base_dim, base_dim, t_dim))

        self.group_norm = nn.GroupNorm(num_groups = 32, num_channels = base_dim)
        self.silu = nn.SiLU()
        self.final_conv = nn.Conv2d(in_channels = base_dim, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, x, t):
        skips = []

        t_emb = self.sin_positional_encoding(t)
        x = self.initial_conv(x)

        x = self.downs[0](x, t_emb)
        skips.append(x)
        x = self.downs[1](x)
        x = self.downs[2](x, t_emb)
        skips.append(x)
        x = self.downs[3](x)
        x = self.downs[4](x, t_emb)
        x = self.downs[5](x)
        skips.append(x)
        x = self.downs[6](x)

        x = self.bottleneck[0](x, t_emb)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t_emb)

        x = self.ups[0](x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.ups[1](x, t_emb)
        x = self.ups[2](x)
        x = self.ups[3](x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.ups[4](x, t_emb)
        x = self.ups[5](x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.ups[6](x, t_emb)

        x = self.silu(self.group_norm(x))

        return self.final_conv(x)
