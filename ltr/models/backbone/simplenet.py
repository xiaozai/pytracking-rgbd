import torch
import torch.nn as nn
from collections import OrderedDict

class SimpleNet(nn.Module):

    def __init__(self, output_layers) -> None:
        super(SimpleNet, self).__init__()
        print('Song in simplenet init')
        self.output_layers = output_layers

        # Inputs : 3 x 288 288
        # layer2 : B, 512, 36, 36
        # layer3 : B 1024, 18, 18


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),               # 64 * 288 * 288
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=3)                               # 64 * 96 * 96
            nn.MaxPool2d(kernel_size=2, stride=2)                               # 64 * 144 * 144
        )

        self.conv2 = nn.Sequential(
            # nn.Conv2d(64, 192, kernel_size=3, stride=3, padding=6),             # 192 * 36 * 36
            nn.Conv2d(64, 192, kernel_size=4, stride=2, padding=1),             # 192 *72 * 72
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                            # 192 * 36 * 36
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 512, kernel_size=3, padding=1),                      # 512 * 36 * 36
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2,  padding=1),           # 1024 * 18 * 18
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, output_layers=None) -> torch.Tensor:

        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if 'layer2' in output_layers:
            outputs['layer2'] = x

        x = self.conv4(x)

        if 'layer3' in output_layers:
            outputs['layer3'] = x

        if len(output_layers) == len(outputs):
            return outputs


        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def simplenet(output_layers=None, pretrained: bool = False, progress: bool = True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            print(l)
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = SimpleNet(output_layers)

    return model
