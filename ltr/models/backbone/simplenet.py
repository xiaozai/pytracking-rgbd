class SimpleNet(nn.Module):

    def __init__(self, output_layers, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.output_layers = output_layers

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, output_layers=None) -> torch.Tensor:

        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers
            print('output_layers : ', output_layers)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        if 'layer3' in output_layers:
            outputs['layer3'] = x

        X = self.conv4(x)
        if 'layer4' in output_layers:
            outputs['layer3'] = x

        if len(output_layers) == len(outputs):
            return outputs


        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def simplenet(output_layers=None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
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

    model = SimpleNet(**kwargs)

    return model
