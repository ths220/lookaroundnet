import torch
import torch.nn as nn
    
class PatchEmbedding(nn.Module):
    def __init__(self, embedding_dim, patch_size, stride_size):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Conv2d(1, embedding_dim, (1, patch_size), (1, stride_size))

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.patcher(out)
        return out
    
class TemporalConvolution(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super(TemporalConvolution, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(nn.Conv2d(embedding_dim, embedding_dim, (1, 3), padding=(0, 1)))
            self.convs.append(nn.ReLU())
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        out = self.convs(x)
        return out
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]
    
class TemporalTransformer(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads, hidden_dim):
        super(TemporalTransformer, self).__init__()

        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True,
                norm_first=True)
            
            self.transformer_layers.append(encoder_layer)

        self.transformer_layers = nn.Sequential(*self.transformer_layers)

    def forward(self, x):
        output = x
        output = self.transformer_layers(output)

        return output
    
class PatchPooling(nn.Module):
    def __init__(self):
        super(PatchPooling, self).__init__()

    def forward(self, x):
        batch_size, num_channels, num_patches, embed = x.shape
        output = x.mean(dim=2).view(batch_size, num_channels, embed)

        return output

class LookAroundNet(nn.Module):
    def __init__(self, num_channels, patch_size, stride_size, embedding_dim, num_heads, num_layers, num_cnn_layers, hidden_dim, num_classes, max_seq_len):
        super(LookAroundNet, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
        self.patch_embedding = PatchEmbedding(embedding_dim, patch_size, stride_size)
        self.temp_conv = TemporalConvolution(embedding_dim, num_cnn_layers)

        self.pos_encoding = LearnablePositionalEncoding(embedding_dim, max_seq_len)
        self.temp_enc = TemporalTransformer(embedding_dim, num_layers, num_heads, hidden_dim)
        
        self.patch_pool = PatchPooling()

        self.channel_pos_encoding = LearnablePositionalEncoding(embedding_dim, num_channels)
        self.channel_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=0.0, batch_first=True)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(embedding_dim * num_channels, num_classes)
    
    def forward(self, x):

        output = self.patch_embedding(x)
        output = self.temp_conv(output)

        output = output.permute(0,2,3,1)
        batch_size, num_channels, num_patches, embed = output.shape
        
        output = output.reshape(batch_size * num_channels, num_patches, embed)
        output = self.pos_encoding(output)

        output = self.temp_enc(output)
        output = output.view(batch_size, num_channels, num_patches, embed)
        output = self.patch_pool(output)
        
        output = self.channel_pos_encoding(output)
        output, _ = self.channel_attention(output, output, output)

        output = output.reshape(batch_size, -1)
        output = self.dropout(output)
        output = self.fc(output)
        
        return output