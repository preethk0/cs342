import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout2d(p = 0.2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout2d(p = 0.2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout2d(p = 0.2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.Dropout2d(p = 0.2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(256, 512, 3, padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.Dropout2d(p = 0.2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.AdaptiveAvgPool2d(1),

            torch.nn.Flatten(),
            torch.nn.Dropout(p = 0.2),
            torch.nn.Linear(512, 6)
        )


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        return self.net(x)


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.Mish(),
            torch.nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.Mish(),
            torch.nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.Mish(),
            torch.nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.Mish(),
            torch.nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.Mish(),
            torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.Mish(),
            torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.Mish(),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Mish(),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.Mish(),
            torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        shape = x.shape
        H = shape[2]
        W = shape[3]
        output = self.net(x)
        output = output[:, :, :H, :W]
        return output


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
