import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation = dilation)
        nn.init.kaiming_uniform_(self.conv1d.weight, nonlinearity = 'linear')

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv1d(x)

class TCN_block(nn.Module):
    def __init__(self, depth=2):    
        super(TCN_block, self).__init__()
        self.depth = depth

        self.Activation_1 = nn.ELU()
        self.TCN_Residual_1 = nn.Sequential(
            CausalConv1d(32, 32, 4, dilation=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            CausalConv1d(32, 32, 4, dilation=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
        )

        self.TCN_Residual = nn.ModuleList()
        self.Activation = nn.ModuleList()
        for i in range(depth - 1):
            TCN_Residual_n = nn.Sequential(
                CausalConv1d(32, 32, 4, dilation=2**(i+1)),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3),
                CausalConv1d(32, 32, 4, dilation=2**(i+1)),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3),
            )
            self.TCN_Residual.append(TCN_Residual_n)
            self.Activation.append(nn.ELU())

    def forward(self, x):
        block = self.TCN_Residual_1(x)
        block += x
        block = self.Activation_1(block)

        for i in range(self.depth - 1):
            block_o = block
            block = self.TCN_Residual[i](block)
            block += block_o
            block = self.Activation[i](block)
        
        return block

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = input_size // (2*num_heads)
        self.num_heads = num_heads
        self.d_k = self.d_v = input_size // (2*num_heads)

        self.W_Q = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_K = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_V = nn.Linear(input_size, self.hidden_size * num_heads)
        self.W_O = nn.Linear(self.hidden_size * num_heads, self.input_size)

        nn.init.normal_(self.W_Q.weight, mean=0.0, std=self.d_k ** -0.5)
        nn.init.normal_(self.W_K.weight, mean=0.0, std=self.d_k ** -0.5)
        nn.init.normal_(self.W_V.weight, mean=0.0, std=self.d_v ** -0.5)
        nn.init.normal_(self.W_O.weight, mean=0.0, std=self.d_v ** -0.5)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2) / (self.hidden_size ** 0.5))
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.dropout(attn_scores)
        attn_output = torch.matmul(attn_scores, V)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output = self.W_O(attn_output)

        return output, attn_scores
    
class attention_block(nn.Module):
    def __init__(self, ):
        super(attention_block, self).__init__()
        self.LayerNorm = nn.LayerNorm(normalized_shape=32, eps=1e-06)
        self.mha = MultiHeadAttention(32, 2)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        att_out, _ = self.mha(x)
        att_out = self.drop(att_out)
        output = att_out.permute(1, 2, 0) + x.permute(1, 2, 0)

        return output

class conv_block(nn.Module):
    def __init__(self, ):
        super(conv_block, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (1, 64), bias = False, padding = 'same'),
            nn.BatchNorm2d(16),
        )
        self.depthwise = nn.Conv2d(16, 16, (22, 1), stride=1, padding=0, dilation=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16*2, 1, 1, 0, 1, 1, bias=False)
        self.conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Conv2d(32, 32, kernel_size=(1, 16), bias = False, padding = 'same'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 7)),
            nn.Dropout(0.5),
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        out = self.conv_block_2(x)
        
        return out
    
class ATCNet(nn.Module):
    def __init__(self, ):
        super(ATCNet, self).__init__()
        self.conv_block = conv_block()
        self.attention_list = nn.ModuleList()
        self.TCN_list = nn.ModuleList()
        self.slideOut_list = nn.ModuleList()
        for i in range(5):
            self.attention_list.append(attention_block())
            self.TCN_list.append(TCN_block())
            self.slideOut_list.append(nn.Linear(32, 4))
        self.out_2 = nn.Linear(160, 4)
    
    def forward(self, x):
        block1 = self.conv_block(x)
        block1 = block1.squeeze(2)
        fuse = 'average'
        n_windows = 5
        sw_concat = []
        for i in range(n_windows):
            st = i
            end = block1.shape[2] - n_windows + i + 1
            block2 = block1[:, :, st: end]
            block2 = self.attention_list[i](block2)
            block3 = self.TCN_list[i](block2)
            block3 = block3[:, :, -1]

            if(fuse == 'average'):
                sw_concat.append(self.slideOut_list[i](block3))
            elif(fuse == 'concat'):
                if i == 0:
                    sw_concat = block3
                else:
                    sw_concat = torch.cat((sw_concat, block3), axis=1)

            if (fuse == 'average'):
                if len(sw_concat) > 1:
                    sw_concat = torch.stack(sw_concat).permute(1, 0, 2)
                    sw_concat = torch.mean(sw_concat, dim=1)
                else:
                    sw_concat = sw_concat[0]
            elif (fuse == 'concat'):
                sw_concat = self.out_2(sw_concat)
            
            return sw_concat

class EEGNet(nn.Module):
    def __init__(self, AF=nn.ELU(alpha=1)):
        super(EEGNet, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, 1e-05)
        )      
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, 1e-05),
            AF,
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, 1e-05),
            AF,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, (7, 8)),
            nn.AvgPool2d(kernel_size=(15, 7), stride=(15, 7))
        )
        
    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.conv(x)
        x = x.squeeze()
        return x