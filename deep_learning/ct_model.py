import torch
import torch.nn as nn 
import torch.nn.functional as F 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import performer_pytorch
from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 256

test_transforms = A.Compose([
    A.Normalize(),
    ToTensorV2()
])
class ModelConfig:
    # Configurations of the Model
    model_type = 'transformer'
    act_type = 'relu'
    enhance_bn = False # To Test
    num_classes = 2 # Often times, CE > BCE.
    
    gate_attention = True # Whether or not to gate attention, makes attention less unstable. 
    attention_type = 'se'
    bottleneck_type = 'inverse'
    transformer_attention = 'performer' # Performer is more memory efficient, faster, etc.
    num_encoders = 2
    reduction = 4
    dilation = 4
    expand = 2
    num_blocks = 3
    max_length = 16 # Not really a hyper parameter, decided based on flattened size after CNN encodings
    transformer_dim = 512

class Mish(pl.LightningModule):
    # Mish Activation Fn
    def __init_(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
def replace_all_act(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.SiLU)):
            setattr(model, name, Mish())
        else:
            replace_all_act(module)
def initialize_weights(layer):
    # Better Initialization of CNN weights.
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming Init
            act_type = ModelConfig.act_type
            nn.init.kaiming_normal_(m.weight, nonlinearity = 'relu')

        elif isinstance(m, nn.BatchNorm2d):
            # 1's and 0's
            m.weight.data.fill_(1)
            m.bias.data.zero_()

# Regular CNN Blocks for BaseLine
class Act(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.act_type = ModelConfig.act_type
        if self.act_type == 'relu':
            self.act = nn.ReLU(inplace = True)
        else:
            self.act = nn.SiLU(inplace = True) # Can be worse, but is faster
    def forward(self, x):
        return self.act(x)
class ConvBlock(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel_size, padding, groups, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, bias = False)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = Act()
        initialize_weights(self)
    def forward(self, x):
        return self.bn(self.act(self.conv(x)))
class EnhancedBN(pl.LightningModule):
    # Enhanced Batch Normalization Block, using Conv2d instead of BN
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.kernel_size = 3
        self.padding = 1
        
        self.enhance_bn = ModelConfig.enhance_bn
        if self.enhance_bn:
            self.bn1 = nn.BatchNorm2d(self.in_features, affine = False)
            self.bn2 = nn.Conv2d(self.in_features, self.in_features, kernel_size = self.kernel_size, padding = self.padding)
        else:
            self.bn1 = nn.BatchNorm2d(self.in_features)
            self.bn2 = nn.Identity()
        initialize_weights(self)
    def forward(self, x):
        return self.bn2(self.bn1(x))
class EnhancedConvBlock(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel_size, padding, groups, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, bias = False)
        self.bn = EnhancedBN(out_features)
        self.act = Act()
        initialize_weights(self)
    def forward(self, x):
        return self.bn(self.act(self.conv(x)))
class SqueezeExcite(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.Act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
    def forward(self, x):
        mean = torch.mean(x, dim = -1)
        mean = torch.mean(mean, dim = -1)
        
        squeeze = self.Act(self.Squeeze(mean))
        excite = torch.sigmoid(self.Excite(squeeze)).unsqueeze(-1).unsqueeze(-1)
        return excite * x
class SpatialAttention(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.Act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
        
        self.Conv = nn.Conv2d(self.in_features, self.inner_features, 1)
        initialize_weights(self)
    def forward(self, x):
        mean = torch.mean(x, dim = -1)
        mean = torch.mean(mean, dim = -1) 
        
        squeeze = self.Act(self.Squeeze(mean))
        excite = torch.sigmoid(self.Excite(squeeze)).unsqueeze(-1).unsqueeze(-1) * x
        
        squeeze_conv = torch.sigmoid(self.Conv(x)) * x
        return (excite + squeeze_conv) / 2
class ECASqueezeExcite(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.kernel_size = 5
        self.padding = 2
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2d = nn.Conv2d(1, 1, kernel_size = self.kernel_size, padding = self.padding, bias = False)
    def forward(self, x):
        pooled = torch.squeeze(self.avgPool(x), dim = -1).transpose(-1, -2) # (B, 1, C)
        conv = torch.sigmoid(self.conv2d(pooled)).transpose(-1, -2).unsqueeze(-1)
        return conv * x
class TransformerSqueezeExcite(pl.LightningModule):
    '''
    Like in T2TVit, SqueezeExcite module for Transformers
    '''
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        
        self.Squeeze = nn.Linear(self.in_features, self.inner_features)
        self.Act = Act()
        self.Excite = nn.Linear(self.inner_features, self.in_features)
        
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        '''
        x: Tensor(B, L, C)
        '''
        squeeze = self.Act(self.Squeeze(x))
        excited = torch.sigmoid(self.Excite(squeeze)) * x
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            return gamma * excited + (1 - gamma) * x
        return excited
        
    
class SelfAttention(pl.LightningModule):
    '''
    Self Attention for ViT
    
    Full O(N^2) Attention: Performer Attention outsourced to pip.
    '''
    def __init__(self, in_features, inner_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.num_heads = num_heads
        
        self.Keys = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Queries = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Values = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Linear = nn.Linear(self.inner_features * self.num_heads, self.in_features)
    def forward(self, x):
        K = self.Keys(x)
        V = self.Queries(x)
        Q = self.Values(x) # (B, L, HI)
        
        K = K.reshape(B, L, self.num_heads, self.inner_features)
        V = V.reshape(B, L, self.num_heads, self.inner_features)
        Q = Q.reshape(B, L, self.num_heads, self.inner_features)
        
        K = K.transpose(1, 2).reshape(B * self.num_heads, L, self.inner_features)
        V = V.transpose(1, 2).reshape(B * self.num_heads, L, self.inner_features)
        Q = Q.transpose(1, 2).reshape(B * self.num_heads, L, self.inner_features)
        
        att_mat = F.softmax(torch.bmm(K, V.transpose(1, 2)) / math.sqrt(self.inner_features))
        att_scores = torch.bmm(att_mat, Q)
        
        att_scores = att_scores.reshape(B, self.num_heads, L, self.inner_features)
        att_scores = att_scores.transpose(1, 2).transpose(B, L, self.num_heads * self.inner_features)
        return self.Linear(att_scores)
class PerformerSelfAttention(pl.LightningModule):
    # Performer O(N) attention, uses fancy matrix operations to perform attention quickly.
    def __init__(self, in_features, inner_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.num_heads = num_heads
        
        self.K = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.V = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        self.Q = nn.Linear(self.in_features, self.inner_features * self.num_heads)
        
        self.Linear = nn.Linear(self.inner_features * self.num_heads, self.in_features)
        self.FastAttention = performer_pytorch.FastAttention(
            dim_heads = self.inner_features, 
            nb_features = self.inner_features * 2
        )
    def forward(self, x):
        B, L, C = x.shape
        K = self.K(x)
        V = self.V(x)
        Q = self.Q(x)
        
        K = K.reshape(B, L, self.num_heads, self.inner_features)
        V = V.reshape(B, L, self.num_heads, self.inner_features)
        Q = Q.reshape(B, L, self.num_heads, self.inner_features)
        
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        Q = Q.transpose(1, 2)
        
        attended = self.FastAttention(Q, K, V)
        
        attended = attended.transpose(1, 2)
        attended = attended.reshape(B, L, self.num_heads, self.inner_features)
        attended = attended.reshape(B, L, self.num_heads * self.inner_features)
        
        return self.Linear(attended)
class TransformerAttention(pl.LightningModule):
    def __init__(self, in_features, inner_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.num_heads = num_heads
        
        self.attention_type = ModelConfig.transformer_attention
        if self.attention_type == 'performer':
            self.layer = PerformerSelfAttention(self.in_features, self.inner_features, self.num_heads)
        else:
            self.layer = SelfAttention(self.in_features, self.inner_features, self.num_heads)
    def forward(self, x):
        return self.layer(x)
class TransformerEncoder(pl.LightningModule):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.reduction = ModelConfig.reduction 
        self.inner_features = self.in_features // self.reduction 
        self.max_length = ModelConfig.max_length 
        
        self.LayerNorm1 = nn.LayerNorm((self.max_length, self.in_features))
        self.Attention = TransformerAttention(self.in_features, self.inner_features, self.reduction)
        self.LayerNorm2 = nn.LayerNorm((self.max_length, self.in_features))
        self.Linear = nn.Linear(self.in_features, self.in_features)
    def forward(self, x):
        norm1 = self.LayerNorm1(x)
        attention = self.Attention(norm1) + x
        
        norm2 = self.LayerNorm2(attention)
        linear = self.Linear(norm2) + attention
        return linear
class Attention(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.attention_type = ModelConfig.attention_type
        assert self.attention_type in ['scse', 'none', 'se', 'eca']
        if self.attention_type == 'eca':
            self.layer = ECASqueezeExcite()
        elif self.attention_type == 'scse':
            self.layer = SpatialAttention(in_features, inner_features)
        elif self.attention_type == 'none':
            self.layer = nn.Identity()
        else:
            self.layer = SqueezeExcite(in_features, inner_features)
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        layer = self.layer(x)
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            return gamma * layer + (1 - gamma) * x
        return layer
class AstrousConvolutionalBlock(pl.LightningModule):
    def __init__(self, in_features, out_features, kernel_size, padding, groups, stride, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups, stride = stride, dilation = dilation, bias = False)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = Act()
        initialize_weights(self)
    def forward(self, x):
        return self.bn(self.act(self.conv(x)))
    
class BAM(pl.LightningModule):
    # BAM modules to be placed in between bottleneck layers in Encoder.
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        
        self.squeeze = nn.Linear(self.in_features, self.inner_features)
        self.act = Act()
        self.excite = nn.Linear(self.inner_features, self.in_features)
        self.dilation = ModelConfig.dilation
        
        self.squeeze_conv = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.dw = AstrousConvolutionalBlock(self.inner_features, self.inner_features, 3, self.dilation, self.inner_features, 1, self.dilation)
        self.excite_conv = ConvBlock(self.inner_features, 1, 1, 0, 1, 1)
        
        self.gate_attention = ModelConfig.gate_attention
        if self.gate_attention:
            self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        mean = torch.mean(x, dim = -1)
        mean = torch.mean(mean, dim = -1)
        
        squeeze = self.act(self.squeeze(mean))
        excite = self.excite(squeeze).unsqueeze(-1).unsqueeze(-1)
        
        squeeze_conv = self.squeeze_conv(x)
        dw = self.dw(squeeze_conv)
        excite_conv = self.excite_conv(dw)
        
        excite = torch.sigmoid((excite + excite_conv) / 2) * x
        if self.gate_attention:
            gamma = torch.sigmoid(self.gamma)
            return gamma * excite + (1 - gamma) * x # gated resnet block.
        return excite + x # Just Simple Add.
        
class BottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.reduction = ModelConfig.reduction
        
        self.Squeeze = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.Process = ConvBlock(self.inner_features, self.inner_features, 3, 1, 1, 1)
        self.Expand = EnhancedConvBlock(self.inner_features, self.in_features, 1, 0, 1, 1)
        self.SE = Attention(self.in_features, self.inner_features // self.reduction)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        squeeze = self.Squeeze(x)
        process = self.Process(squeeze)
        expand = self.Expand(process)
        SE = self.SE(expand)
    
        gamma = torch.sigmoid(self.gamma)
        return gamma * SE + (1 - gamma) * x
class BottleNeckInverse(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.reduction = ModelConfig.reduction
        
        self.Expand = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.DW = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, 1)
        self.SE = Attention(self.inner_features, self.inner_features // self.reduction)
        self.Squeeze = EnhancedConvBlock(self.inner_features, self.in_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        expand = self.Expand(x)
        dw = self.DW(expand)
        se = self.SE(dw)
        squeeze = self.Squeeze(se)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * squeeze + (1 - gamma) * x
class DownSamplerBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride
        self.reduction = ModelConfig.reduction
        
        self.avg_pool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.conv_pool = EnhancedConvBlock(self.in_features, self.out_features, 3, 1, 1, 1)
        
        self.Squeeze = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.Process = ConvBlock(self.inner_features, self.inner_features, 3, 1, 1, 1)
        self.Expand = EnhancedConvBlock(self.inner_features, self.out_features, 1, 0, 1, 1)
        self.SE = Attention(self.out_features, self.out_features // self.reduction)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        pooled = self.avg_pool(x)
        conv_pool = self.conv_pool(pooled)
        
        squeeze = self.Squeeze(pooled)
        process = self.Process(squeeze)
        expand = self.Expand(process)
        SE = self.SE(expand)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * SE + (1  - gamma) * conv_pooled
class DownSamplerBottleNeckInverse(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        
        self.stride = stride
        self.reduction = ModelConfig.reduction
        
        self.avg_pool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.conv_pool = EnhancedConvBlock(self.in_features, self.out_features, 3, 1, 1, 1)
        
        self.Expand = ConvBlock(self.in_features, self.inner_features, 1, 0, 1, 1)
        self.Process = ConvBlock(self.inner_features, self.inner_features, 3, 1, self.inner_features, 1)
        self.SE = Attention(self.inner_features, self.inner_features // self.reduction)
        self.Squeeze = EnhancedConvBlock(self.inner_features, self.out_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        pooled = self.avg_pool(x)
        conv_pool = self.conv_pool(pooled)
        
        expand = self.Expand(pooled)
        process = self.Process(expand)
        se = self.SE(process)
        squeeze = self.Squeeze(se)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * squeeze + (1 - gamma) * conv_pool
class FusedMBConv(pl.LightningModule):
    # Fused MB Conv Blocks
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features 
        self.reduction = ModelConfig.reduction
        
        self.ConvFused = ConvBlock(self.in_features, self.inner_features, 3, 1, 1, 1)
        self.SE = Attention(self.inner_features, self.inner_features // self.reduction)
        self.ConvProj = EnhancedConvBlock(self.inner_features, self.in_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        fused = self.ConvFused(x)
        SE = self.SE(fused)
        proj = self.ConvProj(SE)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * proj + (1 - gamma) * x
        
class DownSamplerFusedMBConv(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        
        self.stride = stride
        self.reduction = ModelConfig.reduction 
        self.pool = nn.AvgPool2d(kernel_size = 3, padding = 1, stride = self.stride)
        self.conv_pool = EnhancedConvBlock(self.in_features, self.out_features, 3, 1, 1, 1)
        
        self.ConvFused = ConvBlock(self.in_features, self.inner_features, 3, 1, 1, 1)
        self.SE = Attention(self.inner_features, self.inner_features // self.reduction)
        self.ConvProj = EnhancedConvBlock(self.inner_features, self.out_features, 1, 0, 1, 1)
        
        self.gamma = nn.Parameter(torch.zeros((1), device = self.device))
    def forward(self, x):
        pooled = self.pool(x)
        conv_pool = self.conv_pool(pooled)
        
        ConvFused = self.ConvFused(pooled)
        SE = self.SE(ConvFused)
        ConvProj = self.ConvProj(SE)
        
        gamma = torch.sigmoid(self.gamma)
        return gamma * ConvProj + (1 - gamma) * conv_pool
class ChooseBottleNeck(pl.LightningModule):
    def __init__(self, in_features, inner_features):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.BottleNeck_Type = ModelConfig.bottleneck_type
        assert self.BottleNeck_Type in ['inverse', 'fused', 'bottleneck', 'none']
        if self.BottleNeck_Type == 'inverse':
            self.layer = BottleNeckInverse(self.in_features, self.inner_features)
        elif self.BottleNeck_Type == 'fused':
            self.layer = FusedMBConv(self.in_features, self.inner_features)
        elif self.BottleNeck_Type == 'bottleneck':
            self.layer = BottleNeck(self.in_featurs, self.inner_features)
    def forward(self, x):
        return self.layer(x)
class ChooseDownSampler(pl.LightningModule):
    def __init__(self, in_features, inner_features, out_features, stride):
        super().__init__()
        self.in_features = in_features
        self.inner_features = inner_features
        self.out_features = out_features
        self.stride = stride
        self.BottleNeck_Type = ModelConfig.bottleneck_type
        assert self.BottleNeck_Type in ['inverse', 'bottleneck', 'fused']
        
        if self.BottleNeck_Type == 'inverse':
            self.layer = DownSamplerBottleNeckInverse(self.in_features, self.inner_features, self.out_features, self.stride)
        elif self.BottleNeck_Type == 'fused':
            self.layer = DownSamplerFusedMBConv(self.in_features, self.inner_features, self.out_features, self.stride)
        else:
            self.layer = DownSamplerBottleNeckInverse(self.in_features, self.inner_features, self.out_features, self.stride)
    def forward(self, x):
        return self.layer(x)
class ViTAlphaBackBone(pl.LightningModule):
    def freeze(self, layers):
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    def unfreeze(self, layers):
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    def __init__(self):
        super().__init__()
        self.model_name = 'efficientnet-b0'
        self.model = EfficientNet.from_name(self.model_name)
        self.encoder_dims = [32, 16, 24, 40, 80, 112, 192, 320, ModelConfig.transformer_dim]
        
        
        self.conv1 = self.model._conv_stem # 32
        self.bn1 = self.model._bn0
        self.act1 = self.model._swish
        
        self.block0 = nn.Sequential(*[self.model._blocks[0]])# 16
        self.block1 = nn.Sequential(*self.model._blocks[1:3]) # 24, downsampled
        self.block2 = nn.Sequential(*self.model._blocks[3:5]) # 40, downsampled
        self.block3 = nn.Sequential(*self.model._blocks[5:8]) # 80, downsampled
        self.block4 = nn.Sequential(*self.model._blocks[8:11]) # 112
        self.block5 = nn.Sequential(*self.model._blocks[11:15]) # 192, downsampled
        self.block6 = nn.Sequential(*self.model._blocks[15:]) # 320
        
        # Freeze Layers
        self.freeze([self.conv1, self.bn1, self.block0, self.block1, self.block2, self.block3, self.block4])
        # Additional Layers
        self.reduction = ModelConfig.reduction
        
        self.Attention6 = BAM(self.encoder_dims[7], self.encoder_dims[7] // self.reduction)
        self.Dropout6 = nn.Dropout2d(0.1)
        
        self.expand = ModelConfig.expand
        self.num_blocks = ModelConfig.num_blocks
        self.block7 = nn.Sequential(*[
            ChooseDownSampler(self.encoder_dims[7], self.encoder_dims[7] * self.expand, self.encoder_dims[8], 2)
        ] + [
            ChooseBottleNeck(self.encoder_dims[8], self.encoder_dims[8] * self.expand) for i in range(self.num_blocks)
        ])
        self.Attention7 = BAM(self.encoder_dims[8], self.encoder_dims[8] // self.reduction)
        self.Dropout7 = nn.Dropout2d(0.1)
    def forward(self, x):
        features0 = self.bn1(self.act1(self.conv1(x))) # 128
        block0 = self.block0(features0) # 128
        block1 = self.block1(block0) # 64
        block2 = self.block2(block1) # 32
        block3 = self.block3(block2) # 16
        block4 = self.block4(block3) # 16
        block5 = self.block5(block4) # 8
        block6 = self.block6(block5) # 8
        
        block6 = self.Dropout6(block6)
        block6 = self.Attention6(block6)
        
        block7 = self.block7(block6)
        block7 = self.Dropout7(block7)
        block7 = self.Attention7(block7) # 4
        return block7
class BaselineHead(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_dim = ModelConfig.transformer_dim
        self.num_classes = ModelConfig.num_classes
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = nn.Linear(self.input_dim, self.num_classes)
    def forward(self, x):
        mean = torch.squeeze(self.global_avg(x))
        return self.Linear(mean)
class ViTAlphaTransformer(pl.LightningModule):
    '''
    Adds a Few Vision Transformer layers on top of the CNN
    '''
    def __init__(self):
        super().__init__()
        self.input_dim = ModelConfig.transformer_dim
        self.num_classes = ModelConfig.num_classes
        self.num_encoders = ModelConfig.num_encoders

        self.encoders = nn.Sequential(*[
            TransformerEncoder(self.input_dim) for i in range(self.num_encoders)
        ])
        self.max_length = ModelConfig.max_length
        self.pos_enc = self.positional_encodings().unsqueeze(0)
        self.Linear = nn.Linear(self.input_dim, self.num_classes)
    def positional_encodings(self):
        import math
        # precomputes the positional_encodings
        L, C = (self.max_length, self.input_dim)
        pos_enc = torch.zeros((L, C), device = self.device)
        for pos in range(L):
            for i in range(0, pos, 2):
                pos_enc[pos, i] = math.sin(pos / 10000 ** (2 * i / self.input_dim))
                pos_enc[pos, i + 1] = math.cos(pos / 10000 ** (2 * (i + 1) / self.input_dim))
        return pos_enc
    def forward(self, x):
        # X: Tensor(B, 512, 4, 4)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-1, -2) # (B, 16, 512)
        # add Positional encodings
        x = x + torch.repeat_interleave(self.pos_enc, B, dim = 0).to(self.device)
        # Encode Using transformers
        features = self.encoders(x) # (B, 16, 512)
        # average over tokens(This is due to how attention is local now using Performer attention, so we want global features)
        mean = torch.mean(features, dim = 1) # (B, 512)
        return self.Linear(mean) # (B, 2)
class ViTAlpha(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.BackBone = ViTAlphaBackBone()
        self.head = BaselineHead() if ModelConfig.model_type == 'baseline' else ViTAlphaTransformer()
        self.use_mish = ModelConfig.act_type == 'mish'
        if self.use_mish:
            replace_all_act(self.BackBone)
    def forward(self, x):
        features = self.BackBone(x)
        return self.head(features)
class TestingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.configure_model()
    def configure_model(self):
        model = ViTAlpha()
        return model
    def forward(self, x):
        self.eval()
        with torch.no_grad():
            pred = F.softmax(self.model(x), dim = -1)
            _, pred = torch.max(pred, dim = -1)
            return pred == 0 # 0 means COVID, 1 means no.
def predict(model, image):
    image = test_transforms(image = image)['image']
    pred = torch.squeeze(model(image.unsqueeze(0))).item()
    return pred
def test(img_path):
    import cv2
    model = load_model()
    image = cv2.imread(img_path)
    return predict(model, image)
def load_model():
    path = './deep_learning/ct.pth'
    model = TestingModel()
    model.load_state_dict(torch.load(path, map_location=device))
    return model