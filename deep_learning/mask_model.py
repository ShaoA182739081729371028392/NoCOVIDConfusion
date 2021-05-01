# This file handles all code relating to the FaceMask Vision Transformer Model.
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import performer_pytorch
from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import math
IMAGE_SIZE = 256
test_transforms = A.Compose([
    A.Normalize(),
    ToTensorV2() 
])
class ModelConfig:
    head_type = 'transformer'
    num_encoder = 3
    num_heads = 4
    
    num_classes = 3
    act = 'relu'
    attention_type = 'scse'
    self_attention_type = 'performer' # More Efficient, Memory Light.
    gate_attention = True # Stabilizes Attention, allowing the model to converge slightly faster
    
    bottleneck_type = 'ghost'
    max_length = 16 # Predetermined, based on the CNN encoder.
    bam_dilate = 3
    reduction = 2
    
    out_dim = 512
    num_blocks = 2
def initialize_weights(layer):
    for module in layer.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            # initialize using Kaiming Normal
            nn.init.kaiming_normal_(module.weight, nonlinearity = 'relu')
        elif isinstance(module, (nn.BatchNorm2d)):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
class Mish(pl.LightningModule):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.tanh(F.softplus(x))
def replace_all_act(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.SiLU)):
            # Replace with Mish
            setattr(module, name, Mish())
        else:
            replace_all_act(module)
class Act(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.act_type = ModelConfig.act
        if self.act_type == 'relu':
            self.act = nn.ReLU(inplace = True)
        elif self.act_type == 'silu':
            self.act = nn.SiLU(inplace = True)
        else:
            self.act = Mish()
    def forward(self, x):
        return self.act(x)
class ConvBlock(pl.LightningModule):
    def __init__(self, in_features,  out_features, kernel_size, padding, groups, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = Act()
        initialize_weights(self)
    def forward(self, x):
        return self.bn(self.act(self.conv(x)))
class SqueezeExcite(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.Act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
    def forward(self, x):
        pool = torch.squeeze(self.avg_pool(x))
        squeeze = self.Act(self.Squeeze(pool))
        excite = torch.sigmoid(self.Excite(squeeze)).unsqueeze(-1).unsqueeze(-1)
        return excite * x
class ECASqueezeExcite(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.kernel_size = 5
        self.padding = 2
        
        self.conv1 = nn.Conv1d(1, 1, kernel_size = self.kernel_size, padding = self.padding, bias = False)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        initialize_weights(self)
    def forward(self, x):
        avg_pool = torch.squeeze(self.avgPool(x), dim = -1).transpose(-1, -2) # (B, 1, C)
        excite = torch.sigmoid(self.conv1(avg_pool)).transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        return excite * x
class SCSE(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features 
        
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.Act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
        
        self.Spatial = nn.Conv2d(self.in_features, 1, kernel_size = 1)
        initialize_weights(self)
    def forward(self, x):
        pooled = torch.squeeze(self.avgPool(x))
        squeeze = self.Act(self.Squeeze(pooled))
        excite = torch.sigmoid(self.Excite(squeeze)).unsqueeze(-1).unsqueeze(-1) * x
        
        excite_conv = torch.sigmoid(self.Spatial(x)) * x
        excited = (excite + excite_conv) / 2
        return excited
class Attention(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.attention_type = ModelConfig.attention_type
        assert self.attention_type in ['eca', 'none', 'se', 'scse']
        if self.attention_type == 'eca':
            self.layer = ECASqueezeExcite()
        elif self.attention_type == 'se':
            self.layer = SqueezeExcite(in_features, inner_features)
        elif self.attention_type == 'scse':
            self.layer = SCSE(in_features, inner_features)
        else:
            self.layer = nn.Identity()
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10) # Init to residual connection at first.
    def forward(self, x):
        val = self.layer(x)
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            return gamma * val + (1 - gamma)
        return val
class MultiHeadedAttention(pl.LightningModule):
    def __init__(self, in_features, inner_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.num_heads = num_heads
    
        self.K = nn.Linear(self.in_features, self.inner_features)
        self.V = nn.Linear(self.in_features, self.inner_features)
        self.Q = nn.Linear(self.in_features, self.inner_features)
        
        self.Linear = nn.Linear(self.in_features, self.inner_features)
    def forward(self, x):
        B, L, C = x.shape
        Keys = self.K(x)
        Values = self.V(x)
        Queries = self.Q(x) # (B, L, HI)
        
        Keys = Keys.reshape(B, L, self.num_heads, self.inner_features)
        Values = Values.reshape(B, L, self.num_heads, self.inner_features)
        Queries = Queries.reshape(B, L, self.num_heads, self.inner_features)
        
        Keys = Keys.transpose(1, 2)
        Values = Values.transpose(1, 2)
        Queries = Queries.transpose(1, 2) # (B, H, L, I)
        
        Keys = Keys.reshape(B * self.num_heads, L, self.inner_features)
        Values = Values.reshape(B * self.num_heads, L, self.inner_features)
        Queries = Queries.reshape(B * self.num_heads, L, self.inner_features) # (BH, L, I)
        
        att_mat = F.softmax(torch.bmm(Keys, Values.transpose(1, 2)) / math.sqrt(self.inner_features))
        att_scores = torch.bmm(att_mat, Queries) #(BH, L, C)
        
        att_scores = att_scores.reshape(B, self.num_heads, L, self.inner_features)
        att_scores = att_scores.transpose(1, 2)
        att_scores = att_scores.reshape(B, L, self.num_heads * self.inner_features)
        return self.Linear(att_scores)
class PerformerAttention(pl.LightningModule):
    def __init__(self, in_features, inner_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.num_heads = num_heads
        
        self.K = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.V = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Q = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Linear = nn.Linear(self.inner_features * self.num_heads, self.in_features)
        
        self.att = performer_pytorch.FastAttention(
            dim_heads = self.inner_features,
            nb_features = self.inner_features
        )
    def forward(self, x):
        B, L, C = x.shape
        Keys = self.K(x)
        Values = self.V(x)
        Queries = self.Q(x) # (B, L, HI)
        
        Keys = Keys.reshape(B, L, self.num_heads, self.inner_features)
        Values = Values.reshape(B, L, self.num_heads, self.inner_features)
        Queries = Queries.reshape(B, L, self.num_heads, self.inner_features)
        
        Keys = Keys.transpose(1, 2)
        Values = Values.transpose(1, 2)
        Queries = Queries.transpose(1, 2) # (B, H, L, self.inner_features)
        
        attended = self.att(Queries, Keys, Values)
        attended = attended.reshape(B, self.num_heads, L, self.inner_features)
        attended = attended.transpose(1, 2)
        attended = attended.reshape(B, L, -1)
        return self.Linear(attended)
class SelfAttention(pl.LightningModule):
    def __init__(self, in_features, inner_features, num_heads):
        super().__init__()
        self.attention_type = ModelConfig.self_attention_type
        if self.attention_type == 'performer':
            self.layer = PerformerAttention(in_features, inner_features, num_heads)
        else:
            self.layer = MultiHeadedAttention(in_features, inner_features, num_heads)
    def forward(self, x):
        return self.layer(x)
class TransformerEncoder(pl.LightningModule):
    def __init__(self, in_features, inner_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.num_heads = num_heads
    
        self.length = ModelConfig.max_length
        self.LayerNorm1 = nn.LayerNorm((self.length, self.in_features))
        self.SA = SelfAttention(self.in_features, self.inner_features, self.num_heads)
        self.LayerNorm2 = nn.LayerNorm((self.length, self.in_features))
        self.Linear = nn.Linear(self.in_features, self.in_features)
    def forward(self, x):
        norm1 = self.LayerNorm1(x)
        SA = self.SA(norm1) + x 
        norm2 = self.LayerNorm2(SA)
        linear = self.Linear(norm2) + SA
        return linear
        
class AstrousConvBlock(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel_size, padding, groups, stride, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, dilation = dilation)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = Act()
        initialize_weights(self)
    def forward(self, x):
        return self.bn(self.act(self.conv(x)))
class BAM(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.bam_dilate = ModelConfig.bam_dilate
    
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
        
        self.Squeeze_Conv = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.DA = AstrousConvBlock(self.inner_features, self.inner_features, 3, self.bam_dilate, self.inner_features, 1, self.bam_dilate)
        self.Excite_Conv = ConvBlock(self.inner_features, 1, 1, 0, 1, 1)
        
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        pooled = torch.squeeze(self.avgPool(x))
        squeeze = self.act(self.Squeeze(pooled))
        excite = torch.sigmoid(self.Excite(squeeze)).unsqueeze(-1).unsqueeze(-1) * x
        
        squeeze_conv = self.Squeeze_Conv(x)
        DA = self.DA(squeeze_conv)
        excite_conv = torch.sigmoid(self.Excite_Conv(DA)) 
        excite_conv = excite_conv * x
        excited = (excite + excite_conv) / 2
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            return excited * gamma + (1 - gamma) * x
        return excited
class SplitAttention(pl.LightningModule):
    # Basic Implementation of Split Attention in ResNeSt
    def __init__(self, in_features, inner_features, cardinality):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.cardinality = cardinality
        
        assert self.inner_features % self.cardinality == 0
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.Squeeze = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.Excite = nn.Conv2d(self.inner_features, self.in_features * self.cardinality, kernel_size =1, groups = self.cardinality)
        
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
        initialize_weights(self)
    def forward(self, x):
        '''
        x: Tensor(B, C, H, W, Cardinality), where Cardinality is the number of groups to apply split attention to.
        '''
        B, C, H, W, Cardinality = x.shape
        assert Cardinality == self.cardinality
        # Sum across all groups
        summed = torch.sum(x, dim = -1) # (B, C, H, W)
        # Pool
        pooled = self.global_pool(summed) # (B, C, 1, 1)
        # Conv
        squeeze = self.Squeeze(pooled) # (B, I, 1, 1)
        excite = torch.squeeze(self.Excite(squeeze)) # (B, Cardinality * Channels)
        
        excite = F.softmax(excite.reshape(B, C, Cardinality), dim = -1) 
        excite = excite.unsqueeze(2).unsqueeze(2) # (B, C, 1, 1, Cardinality)
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            excited = excite * x * gamma + (1 - gamma) * x
        else:
            excited = excite * x # (B, C, H, W, Cardinality)
        
        excited = torch.sum(excited, dim = -1) # (B, C, H, W)
        return excited
class ResNext(pl.LightningModule):
    '''
    ResNext Block. I would make ResNest, but that is just ResNext + SplitAttention(implemented above)
    '''
    def __init__(self, in_features, inner_features, cardinality):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.cardinality = cardinality
        self.reduction = ModelConfig.reduction
        
        self.Squeeze = ConvBlock(self.in_features, self.inner_features * self.cardinality, 1, 0, 1, 1)
        self.Process = ConvBlock(self.inner_features * self.cardinality, self.inner_features * self.cardinality, 3, 1, self.cardinality, 1)
        self.Expand = ConvBlock(self.inner_features * self.cardinality, self.in_features, 1, 0, 1, 1)
        self.SE = Attention(self.in_features, self.in_features // self.reduction)
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        squeeze = self.Squeeze(x)
        process = self.Process(squeeze)
        expand = self.Expand(process)
        SE = self.SE(expand)
        gamma = torch.sigmoid(self.gamma)
        return gamma * SE + (1 - gamma) * x
class BottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.reduction = ModelConfig.reduction
        
        self.Squeeze = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.Process = ConvBlock(self.inner_features, self.inner_features, 3, 1, 1, 1)
        self.Expand = ConvBlock(self.inner_features, self.in_features, 1, 0, 1, 1)
        self.SE = Attention(self.in_features, self.in_features // self.reduction)
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        
        squeeze = self.Squeeze(x)
        process = self.Process(squeeze)
        expand = self.Expand(process)
        excited = self.SE(expand) 
        
        gamma = torch.sigmoid(excited)
        return excited * gamma + (1 - gamma) * x
    
class DownsamplerBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride
        self.reduction = ModelConfig.reduction
        
        self.avgPool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.ConvAvg = ConvBlock(self.in_features, self.out_features, 1, 0, 1, 1)
    
        self.Squeeze = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.Process = ConvBlock(self.inner_features, self.inner_features, 3, 1, 1, self.stride)
        self.Expand = ConvBlock(self.inner_features, self.out_features, 1, 0, 1, 1)
        self.SE = Attention(self.out_features, self.out_features // self.reduction)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        pooled = self.avgPool(x)
        conv_pool = self.ConvAvg(pooled)
        
        squeeze = self.Squeeze(x)
        process = self.Process(squeeze)
        expand = self.Expand(process)
        SE = self.SE(expand)
        
        gamma = torch.sigmoid(self.gamma)
        return SE * gamma + (1 - gamma) * conv_pool
class InverseBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.reduction = ModelConfig.reduction
        
        self.Expand = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.DW = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, 1)
        self.SE = Attention(self.inner_features, self.inner_features // self.reduction)
        self.Squeeze = ConvBlock(self.inner_features, self.in_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        expand = self.Expand(x)
        DW = self.DW(expand)
        SE = self.SE(DW)
        squeeze = self.Squeeze(SE)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * squeeze + (1 - gamma) * x
class DownsamplerInverseBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features 
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride
        self.reduction = ModelConfig.reduction
        
        self.AvgPool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.ConvPool = ConvBlock(self.in_features, self.out_features, 1, 0, 1, 1)
    
        self.Expand = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.DW = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, self.stride)
        self.SE = Attention(self.inner_features, self.inner_features // self.reduction)
        self.Squeeze = ConvBlock(self.inner_features, self.out_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        pooled = self.AvgPool(x)
        convPool = self.ConvPool(pooled)
        
        expand = self.Expand(x)
        DW = self.DW(expand)
        SE = self.SE(DW)
        squeeze = self.Squeeze(SE)
        
        gamma = torch.sigmoid(self.gamma) 
        return gamma * squeeze + (1 - gamma) * convPool
class GhostConvBlock(pl.LightningModule):
    # Ghost ConvBlock - Good Performance for Little Params
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert self.out_features % 2 == 0
        self.inner_features = self.out_features // 2
        
        self.Squeeze = nn.Conv2d(self.in_features, self.inner_features, kernel_size = 1)
        self.DW = nn.Conv2d(self.inner_features, self.inner_features, kernel_size = 1, groups = self.inner_features)
        self.BN = nn.BatchNorm2d(self.inner_features * 2)
        self.act = Act()
    def forward(self, x):
        squeeze = self.Squeeze(x)
        DW = self.DW(squeeze)
        concat = torch.cat([squeeze, DW], dim = 1)
        return self.BN(self.act(concat))
class GhostBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.reduction = ModelConfig.reduction
        
        self.Conv1 = GhostConvBlock(self.in_features, self.inner_features)
        self.Conv2 = GhostConvBlock(self.inner_features, self.in_features)
        self.Attention = Attention(self.in_features, self.in_features // self.reduction)
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        conv1 = self.Conv1(x)
        conv2 = self.Conv2(conv1)
        conv2 = self.Attention(conv2)
        gamma = torch.sigmoid(self.gamma)
        return gamma * conv2 + (1 - gamma) * x
class DownsamplerGhostBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features= out_features
        self.stride = stride
        self.reduction = ModelConfig.reduction
        
        self.AvgPool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.GhostPool = GhostConvBlock(self.in_features, self.out_features)
        
        self.Ghost1 = GhostConvBlock(self.in_features, self.inner_features)
        self.DW = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, self.stride)
        self.Ghost2 = GhostConvBlock(self.inner_features, self.out_features)
        self.Attention = Attention(self.out_features, self.out_features // self.reduction)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device) - 10)
    def forward(self, x):
        pooled = self.AvgPool(x)
        ghost_pool = self.GhostPool(pooled)
        
        ghost_1 = self.Ghost1(x)
        dw = self.DW(ghost_1)
        ghost_2 = self.Ghost2(dw)
        attention = self.Attention(ghost_2)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * attention + (1 - gamma) * ghost_pool
        
class ChooseBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.bottleneck_type = ModelConfig.bottleneck_type
        assert self.bottleneck_type in ['ghost', 'inverse', 'bottleneck']
        if self.bottleneck_type == 'ghost':
            self.layer = GhostBottleNeck(in_features, inner_features)
        elif self.bottleneck_type == 'inverse':
            self.layer = InverseBottleNeck(in_features, inner_features)
        else:
            self.layer = BottleNeck(in_features, inner_features)
    def forward(self, x):
        return self.layer(x)
class ChooseDownsampler(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.bottleneck_type = ModelConfig.bottleneck_type
        assert self.bottleneck_type in ['ghost', 'inverse', 'bottleneck']
        if self.bottleneck_type == 'ghost':
            self.layer = DownsamplerGhostBottleNeck(in_features, inner_features, out_features, stride)
        elif self.bottleneck_type == 'inverse':
            self.layer = DownsamplerInverseBottleNeck(in_features, inner_features, out_features, stride)
        else:
            self.layer = DownsamplerBottleNeck(in_features, inner_features, out_features, stride)
    def forward(self, x):
        return self.layer(x)
class FeatureExtractor(pl.LightningModule):
    def freeze(self, layers):
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    def __init__(self):
        super().__init__()
        self.model_name = 'efficientnet-b0'
        self.model = EfficientNet.from_name(self.model_name)
        # Extract Layers
        self.conv1 = self.model._conv_stem
        self.bn1 = self.model._bn0
        self.act1 = self.model._swish
        self.enc_dims = [3, 32, 24, 40, 80, 112, 192, 320]
        
        self.block0 = self.model._blocks[0]
        self.block1 = nn.Sequential(*self.model._blocks[1:3])
        self.block2 = nn.Sequential(*self.model._blocks[3:5])
        self.block3 = nn.Sequential(*self.model._blocks[5:8])
        self.block4 = nn.Sequential(*self.model._blocks[8:11])
        self.block5 = nn.Sequential(*self.model._blocks[11:15])
        self.block6 = self.model._blocks[15]
        
        self.freeze([self.conv1, self.bn1, self.block0, self.block1, self.block2, self.block3, self.block4])
        # Custom Layers
        self.reduction = ModelConfig.reduction
        
        self.Attention6 = BAM(self.enc_dims[-1], self.enc_dims[-1] // self.reduction)
        self.Dropout6 = nn.Dropout2d(0.1)
        
        self.out_dim = ModelConfig.out_dim
        self.num_blocks = ModelConfig.num_blocks
        self.block7 = nn.Sequential(*[
            ChooseDownsampler(self.enc_dims[-1], self.enc_dims[-1] // self.reduction, self.out_dim, 2)
        ] + [
            ChooseBottleNeck(self.out_dim, self.out_dim // self.reduction) for i in range(self.num_blocks)
        ])
        
        self.Attention7 = BAM(self.out_dim, self.out_dim // self.reduction)
        self.Dropout7 = nn.Dropout2d(0.1)
    
    def forward(self, x):
        features0 = self.bn1(self.act1(self.conv1(x))) # (B, 32, 128, 128)
        block0 = self.block0(features0) # (b, 16, 128, 128)
        block1 = self.block1(block0) # (B, 24, 64, 64)
        block2 = self.block2(block1) # (B, 40, 32, 32)
        block3 = self.block3(block2) # (B, 80, 16, 16)
        block4 = self.block4(block3) # (b, 112, 16, 16)
        block5 = self.block5(block4) # (B, 192, 8, 8)
        block6 = self.block6(block5) # (B, 320, 8, 8)
        # Custom Layer
        block6 = self.Attention6(self.Dropout6(block6))
        
        block7 = self.block7(block6)
        block7 = self.Attention7(self.Dropout7(block7))
        
        return block7 # (B, 512, 4, 4)
class BaseLineHead(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = ModelConfig.num_classes
        self.out_dim = ModelConfig.out_dim
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = nn.Linear(self.out_dim, self.num_classes)
    def forward(self, x):
        avg = torch.squeeze(self.global_avg(x))
        return self.Linear(avg)
class ViTAlphaHead(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = ModelConfig.num_classes
        self.out_dim = ModelConfig.out_dim
        self.num_encoder = ModelConfig.num_encoder
        self.max_length = ModelConfig.max_length
        self.num_heads = ModelConfig.num_heads
        self.positional_encodings = self.pos_enc().to(self.device)
        
        self.encoders = nn.Sequential(*[
            TransformerEncoder(self.out_dim, self.out_dim // self.num_heads, self.num_heads) for i in range(self.num_encoder)
        ])
        
        self.Linear = nn.Linear(self.out_dim, self.num_classes) 
    def pos_enc(self):
        L, C = self.max_length, self.out_dim
        pos_enc = torch.zeros((L, C), device = self.device)
        for pos in range(L):
            for i in range(0, C, 2):
                pos_enc[pos, i] = math.sin(pos / 10000 ** (2 * i / self.out_dim))
                pos_enc[pos, i + 1] = math.cos(pos / 10000 ** (2 * (i + 1) / self.out_dim))
        return pos_enc 
    def forward(self, x):
        B, C, H, W = x.shape
        assert H * W == self.max_length
        # Flatten
        flat_input = x.reshape(B, C, H * W).transpose(1, 2) # (B, L, C)
        # Add Positional Encodings
        positional = torch.repeat_interleave(self.positional_encodings.unsqueeze(0), B, dim = 0).to(self.device) + flat_input
        # Encode using transformers
        encoded = self.encoders(positional)
        # Avg Pool
        pooled = torch.mean(encoded, dim = 1)
        return self.Linear(pooled)
class ViTAlpha(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.head_type = ModelConfig.head_type
        self.head = BaseLineHead() if self.head_type == 'baseline' else ViTAlphaHead()
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)
class TestingModelAlpha(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.configure_model()
    def configure_model(self):
        model = ViTAlpha()
        return model
    def forward(self, x):
        self.eval()
        with torch.no_grad():
            pred = torch.sigmoid(self.model(x))
            return pred > 0.5
def decode(pred):
    present = []
    if pred[0] == 1: #Mouth, Chin, Nose
        present += ['Mouth']
    if pred[1] == 1:
        present += ['Chin']
    if pred[2] == 1:
        present += ['Nose']
    return present
def predict(model, image):
    image = test_transforms(image = image)['image'].unsqueeze(0)
    pred = torch.squeeze(model(image))
    return decode(pred)
def load_model():
    # Loads in the State Dict and Model
    path = './deep_learning/masks.pth'
    model = TestingModelAlpha()
    model.load_state_dict(torch.load(path, map_location=device))
    return model