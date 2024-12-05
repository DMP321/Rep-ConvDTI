import torch
import torch.nn as nn
import torch.nn.functional as F


class Gated_Atten(nn.Module):
    def __init__(self, channels, alpha=0.5):
        super(Gated_Atten, self).__init__()
        self.alpha = alpha
        self.attention_layer_down = nn.Linear(channels, channels//2)
        self.attention_layer_up = nn.Linear(channels//2, channels)
        self.protein_attention_layer = nn.Linear(channels, channels)
        self.drug_attention_layer = nn.Linear(channels, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, y):
        drug_att = self.drug_attention_layer(x.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(y.permute(0, 2, 1))

        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, y.shape[-1], 1)  # repeat along protein size
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, x.shape[-1], 1, 1)  # repeat along drug size
        Atten_matrix = self.attention_layer_up(self.relu(self.attention_layer_down(self.relu(d_att_layers + p_att_layers))))
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))*self.tanh(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))*self.tanh(Protein_atte.permute(0, 2, 1))

        drugConv = x * self.alpha + x * Compound_atte
        proteinConv = y * self.alpha + y * Protein_atte
        return drugConv, proteinConv

class Layer_SE(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(Layer_SE, self).__init__()
        self.down = nn.Linear(input_channels, internal_neurons)
        self.up = nn.Linear(internal_neurons, input_channels)

        self.input_channels = input_channels
        self.nonlinear = nn.GELU()


    def forward(self, inputs):
        x = F.adaptive_avg_pool1d(inputs, output_size=1).permute(0, 2, 1)
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs*x.view(-1, self.input_channels, 1)
    
class DilatedReparamConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DilatedReparamConv, self).__init__()
        self.origin_cnn = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, dilation=1)
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]

        else:
            raise ValueError('Dilated Reparam Block requires kernel_size <= 17')
        
        self.origin_ln = nn.LayerNorm(out_channels)

        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=1,
                                        padding=(r * (k - 1) + 1) // 2, dilation=r))
            self.__setattr__('dil_ln_k{}_{}'.format(k, r), nn.LayerNorm(out_channels))


    def forward(self, x):
        out = self.origin_ln(self.origin_cnn(x).permute(0, 2, 1)).permute(0, 2, 1)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            ln = self.__getattr__('dil_ln_k{}_{}'.format(k, r))
            out = out + ln(conv(x).permute(0, 2, 1)).permute(0, 2, 1)
        return out

class Conv_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv_1D, self).__init__()
        self.conv_1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.layer_norm = nn.LayerNorm(out_channels)
    
    def forward(self, inputs):
        x = self.conv_1d(inputs).permute(0, 2, 1)
        return self.layer_norm(x).permute(0, 2, 1)


class LORS_Block(nn.Module): # LGCNN or SMCNN Block
    def __init__(self, in_channels, out_channels, kernel_size, reparam = False):
        super(LORS_Block, self).__init__()
        if kernel_size > 7 and reparam:
            self.conv1d = DilatedReparamConv(in_channels, out_channels, kernel_size)
        else:
            self.conv1d = Conv_1D(in_channels, out_channels, kernel_size)
        self.gelu = nn.GELU()
        self.se = Layer_SE(out_channels, out_channels//2)

    def forward(self, inputs):
        x = self.conv1d(inputs)
        x = F.gelu(x)
        x = self.se(x)
        x = F.layer_norm(x, [x.shape[-1]])
        return x


class Rep_ConvDTI(nn.Module):
    def __init__(self,config):
        super(Rep_ConvDTI, self).__init__()
        drug_MAX_LENGH = config.drug_len
        protein_MAX_LENGH = config.prot_len

        drug_kernel = eval(config.drug_kernel)
        prot_kernel = eval(config.prot_kernel)
        drug_channel = eval(config.drug_channel)
        prot_channel = eval(config.prot_channel)
        prot_embedding = config.prot_embedding
        drug_embedding = config.drug_embedding
        self.reparam = config.reparam
        self.drug_embeding = nn.Embedding(65, drug_embedding, padding_idx=0)
        self.protein_embeding = nn.Embedding(26, prot_embedding, padding_idx=0)
        self.drug_CNN = nn.Sequential(
            *[LORS_Block(drug_channel[i], drug_channel[i+1], kernel_size=drug_kernel[i], reparam=self.reparam) for i in range(len(drug_channel)-1)])
        self.protein_CNN = nn.Sequential(
            *[LORS_Block(prot_channel[i], prot_channel[i+1], kernel_size=prot_kernel[i], reparam=self.reparam) for i in range(len(prot_channel)-1)])
        self.gated_att = Gated_Atten(drug_channel[-1])
        self.Drug_max_pool = nn.MaxPool1d(drug_MAX_LENGH)
        self.Protein_max_pool = nn.MaxPool1d(protein_MAX_LENGH)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(drug_channel[-1] + prot_channel[-1], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
        

    def fussion(self, drug, prot):
        
        drug_embed = self.drug_embeding(drug)
        prot_embed = self.protein_embeding(prot)
        drug_embed = drug_embed.permute(0, 2, 1)
        prot_embed = prot_embed.permute(0, 2, 1)
        
        drug_x = self.drug_CNN(drug_embed)

        protein_x = self.protein_CNN(prot_embed)
        
        x, y = self.gated_att(drug_x, protein_x)
        
        drugConv = self.Drug_max_pool(x).squeeze(2)
        proteinConv = self.Protein_max_pool(y).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)

        return self.fc1(pair)

    def forward(self, drug, prot):
        
        drug_embed = self.drug_embeding(drug)
        prot_embed = self.protein_embeding(prot)
        drug_embed = drug_embed.permute(0, 2, 1)
        prot_embed = prot_embed.permute(0, 2, 1)
        
        drug_x = self.drug_CNN(drug_embed)

        protein_x = self.protein_CNN(prot_embed)

        x, y = self.gated_att(drug_x, protein_x)
        
        drugConv = self.Drug_max_pool(x).squeeze(2)
        proteinConv = self.Protein_max_pool(y).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.gelu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.gelu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.gelu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict.squeeze(1)

