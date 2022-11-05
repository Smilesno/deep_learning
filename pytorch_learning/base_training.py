import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from MyModel import MyModel

train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=False,
    transform=ToTensor()
)

train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(model, dataloder, loss_fn, optimizer):
    model.train()
    for batch, (X_train, y) in enumerate(dataloder):
        X_train, y = X_train.to(device), y.to(device)
        outputs = model(X_train)
        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(dataloder)
            print(f"loss: {loss}, current: {current}")

def test(model, dataloder, loss_fn):
    model.eval()
    loss, accuracy = 0, 0
    with torch.no_grad():
        for X_test, y in dataloder:
            X_test, y = X_test.to(device), y.to(device)
            outputs = model(X_test)
            loss += loss_fn(outputs, y).item()
            accuracy += (outputs.argmax(1) == y).type(torch.float).sum().item()

    loss /= len(dataloder)
    accuracy /= len(dataloder.dataset)

    print(f"test loss: {loss}")
    print(f"accuracy: {accuracy}")

epoch = 5
for i in range(epoch):
    print(f"===============epoch {i+1} ================")
    train(model, train_data_loader, loss_fn, optimizer)
    test(model, test_data_loader, loss_fn)

print("done")

torch.save(model.state_dict(), "model.pth")
print("Saved Pytorch Model State to model.pth")


## load model
# model = MyModel()
# model.load_state_dict(torch.load("model.pth"))


