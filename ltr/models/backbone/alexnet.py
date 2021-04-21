class AlexNet(nn.Module):

    def __init__(self, output_layers, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.output_layers = output_layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        self.layer_convert = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_layers=None) -> torch.Tensor:

        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers
            print('output_layers : ', output_layers)
        x = self.features(x)
        print(x.shape)
        x = self.avgpool(x)
        #
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        y = self.layer_convert(x)

        outputs['layer3'] = x
        outputs['layer4'] = y

        return outputs


        raise ValueError('output_layer is wrong.')


def alexnet(output_layers=None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
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

    model = AlexNet(**kwargs)
    if pretrained:
        print('loading pretrained Alexnet ...')
        pretrained_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # model.load_state_dict(state_dict)
    return model
