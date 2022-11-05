from torch.nn import *

class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = Sequential(
            Conv2d(1, 18, kernel_size=5),
            MaxPool2d(2),
            Conv2d(18, 36, kernel_size=3),
            MaxPool2d(2),
            Flatten(),
            Linear(900, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
