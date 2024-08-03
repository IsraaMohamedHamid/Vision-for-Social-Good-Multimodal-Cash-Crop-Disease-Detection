import torch
from torch import Tensor
import torch.nn as nn
from torchvision import models
from torchvision.models.efficientnet import MBConv
from typing import Type, Any, Callable, Union, List, Optional

from AACN_Layer import AACN_Layer, AACN_EfficientNet_Layer, AACN_VGG_Layer

__all__ = ['ResNet', 'attention_augmented_resnet18', 'attention_augmented_resnet34', 'attention_augmented_resnet50', 'attention_augmented_resnet101',
           'attention_augmented_resnet152', 'attention_augmented_resnext50_32x4d', 'attention_augmented_resnext101_32x8d',
           'wide_attention_augmented_resnet50_2', 'wide_attention_augmented_resnet101_2']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = False,
        num_heads: int = 8,
        k: float = 0.25,
        v: float = 0.25,
        image_size: int = 224,
        inference: bool = False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if not attention:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = AACN_Layer(in_channels=planes, k=k, v=v, kernel_size=3, num_heads=num_heads, image_size=image_size, inference=inference)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:attention_augmented_resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = False,
        num_heads: int = 8,
        k: float = 0.25,
        v: float = 0.25,
        image_size: int = 224,
        inference: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride)
        self.bn1 = norm_layer(width)
        if not attention:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = AACN_Layer(in_channels=width, k=k, v=v, kernel_size=3, num_heads=num_heads, image_size=image_size, inference=inference)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: List[bool] = [False, False, False, False],
        num_heads: int = 8,
        k: float = 0.25,
        v: float = 0.25,
        image_size: int = 224,
        inference: bool = False
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], attention=attention[0], num_heads=num_heads, k=k, v=v, image_size=image_size//4, inference=inference)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], attention=attention[1], num_heads=num_heads, k=k, v=v, image_size=image_size//8, inference=inference)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], attention=attention[2], num_heads=num_heads, k=k, v=v, image_size=image_size//16, inference=inference)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], attention=attention[3], num_heads=num_heads, k=k, v=v, image_size=image_size//32, inference=inference)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1, dilate: bool = False, attention: bool = False, num_heads: int = 8, k: float = 0.25, v: float = 0.25, image_size: int = 224, inference: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, attention, num_heads=num_heads, k=k, v=v, image_size=image_size, inference=inference))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attention=attention, num_heads=num_heads, k=k, v=v, image_size=image_size, inference=inference))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _attention_augmented_resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def attention_augmented_resnet18(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _attention_augmented_resnet('attention_augmented_resnet18', BasicBlock, [2, 2, 2, 2], progress,
                   **kwargs)


def attention_augmented_resnet34(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _attention_augmented_resnet('attention_augmented_resnet34', BasicBlock, [3, 4, 6, 3], progress,
                   **kwargs)


def attention_augmented_resnet50(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _attention_augmented_resnet('attention_augmented_resnet50', Bottleneck, [3, 4, 6, 3], progress,
                   **kwargs)


def attention_augmented_resnet101(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _attention_augmented_resnet('attention_augmented_resnet101', Bottleneck, [3, 4, 23, 3], progress,
                   **kwargs)


def attention_augmented_resnet152(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _attention_augmented_resnet('attention_augmented_resnet152', Bottleneck, [3, 8, 36, 3], progress,
                   **kwargs)


def attention_augmented_resnext50_32x4d(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _attention_augmented_resnet('attention_augmented_resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   progress, **kwargs)


def attention_augmented_resnext101_32x8d(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _attention_augmented_resnet('attention_augmented_resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   progress, **kwargs)


def wide_attention_augmented_resnet50_2(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _attention_augmented_resnet('wide_attention_augmented_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   progress, **kwargs)


def wide_attention_augmented_resnet101_2(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _attention_augmented_resnet('wide_attention_augmented_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   progress, **kwargs)

class AttentionAugmentedEfficientNet(nn.Module):
    def __init__(self, attention: bool = False, num_heads: int = 8, k: float = 0.25, v: float = 0.25, image_size: int = 224, inference: bool = False):
        super(AttentionAugmentedEfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        if attention:
            # Modify the layers with attention
            self.modify_attention_layers(num_heads, k, v, image_size, inference)

class AttentionAugmentedEfficientNet(nn.Module):
    def __init__(self, attention: bool = False, num_heads: int = 8, k: float = 0.25, v: float = 0.25, image_size: int = 224, inference: bool = False):
        super(AttentionAugmentedEfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        if attention:
            # Modify the layers with attention
            self.modify_attention_layers(num_heads, k, v, image_size, inference)

    def modify_attention_layers(self, num_heads, k, v, image_size, inference):
        def replace_conv(layer, in_channels, out_channels):
            return nn.Sequential(
                AACN_EfficientNet_Layer(in_channels=in_channels, k=k, v=v, kernel_size=3, num_heads=num_heads, image_size=image_size, inference=inference),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        # EfficientNet-B0 layers that should have attention augmentation
        # Based on the EfficientNet architecture, these layers are blocks that need their conv2d replaced
        # Adjust the layers to ensure channel alignment
        self.efficientnet.features[1][0].block[0] = replace_conv(self.efficientnet.features[1][0].block[0], in_channels=32, out_channels=16)
        self.efficientnet.features[2][0].block[0] = replace_conv(self.efficientnet.features[2][0].block[0], in_channels=16, out_channels=24)
        self.efficientnet.features[3][0].block[0] = replace_conv(self.efficientnet.features[3][0].block[0], in_channels=24, out_channels=40)
        self.efficientnet.features[4][0].block[0] = replace_conv(self.efficientnet.features[4][0].block[0], in_channels=40, out_channels=80)
        self.efficientnet.features[5][0].block[0] = replace_conv(self.efficientnet.features[5][0].block[0], in_channels=80, out_channels=112)
        self.efficientnet.features[6][0].block[0] = replace_conv(self.efficientnet.features[6][0].block[0], in_channels=112, out_channels=192)
        self.efficientnet.features[7][0].block[0] = replace_conv(self.efficientnet.features[7][0].block[0], in_channels=192, out_channels=320)

    def forward(self, x: Tensor) -> Tensor:
        return self.efficientnet(x)

# Function to create an instance of the modified EfficientNet
def attention_augmented_efficientnetb0(attention: bool = False, **kwargs: any) -> AttentionAugmentedEfficientNet:
    return AttentionAugmentedEfficientNet(attention=attention, **kwargs)

class AttentionAugmentedInceptionV3(nn.Module):
    def __init__(self, attention: bool = False, num_heads: int = 8, k: float = 0.25, v: float = 0.25, image_size: int = 299, inference: bool = False):
        super(AttentionAugmentedInceptionV3, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        if attention:
            self.inception.Mixed_5b.branch3x3_2 = AACN_Layer(in_channels=96, k=k, v=v, kernel_size=3, num_heads=num_heads, image_size=image_size//16, inference=inference)
            self.inception.Mixed_5c.branch3x3_2 = AACN_Layer(in_channels=128, k=k, v=v, kernel_size=3, num_heads=num_heads, image_size=image_size//16, inference=inference)
        
        # Adjust the auxiliary classifier
        if self.inception.aux_logits:
            self.inception.AuxLogits.conv0 = nn.Conv2d(768, 128, kernel_size=1)
            self.inception.AuxLogits.conv1 = nn.Conv2d(128, 768, kernel_size=3, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.inception(x)


class AttentionAugmentedViT(nn.Module):
    def __init__(self, attention: bool = False, num_heads: int = 8, k: float = 0.25, v: float = 0.25, image_size: int = 224, inference: bool = False):
        super(AttentionAugmentedViT, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        if attention:
            self.vit.encoder.ln = AACN_Layer(in_channels=768, k=k, v=v, kernel_size=1, num_heads=num_heads, image_size=image_size//16, inference=inference)

    def forward(self, x: Tensor) -> Tensor:
        return self.vit(x)


def attention_augmented_efficientnetb0(attention: bool = False, **kwargs: Any) -> AttentionAugmentedEfficientNet:
    r""" EfficientNet model from 
    """
    return AttentionAugmentedEfficientNet(attention=attention, **kwargs)


def attention_augmented_inceptionv3(attention: bool = False, **kwargs: Any) -> AttentionAugmentedInceptionV3:
    r""" Inceptionv3 model from 
    """
    return AttentionAugmentedInceptionV3(attention=attention, **kwargs)


def attention_augmented_vit(attention: bool = False, **kwargs: Any) -> AttentionAugmentedViT:
    r""" ViT model from 
    """
    return AttentionAugmentedViT(attention=attention, **kwargs)

class AttentionAugmentedVGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(AttentionAugmentedVGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def attention_augmented_vgg(model_name='VGG16', num_classes=4, init_weights=True):
    cfgs = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def _make_layers(cfg, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        layers += [AACN_VGG_Layer(in_channels, in_channels, kernel_size=3, num_heads=8)]
        return nn.Sequential(*layers)

    assert model_name in cfgs, f"Model {model_name} not recognized. Available models: {list(cfgs.keys())}"
    cfg = cfgs[model_name]
    features = _make_layers(cfg)
    model = AttentionAugmentedVGG(features, num_classes=num_classes, init_weights=init_weights)
    return model