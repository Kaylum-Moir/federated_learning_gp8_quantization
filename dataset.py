from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_datasets(num_clients):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    data_size = len(trainset) // num_clients

    trainloaders = []

    for i in range(num_clients):

        indices = list(range(i * data_size, (i + 1) * data_size))

        subset = Subset(trainset, indices)

        trainloader = DataLoader(subset, batch_size=32, shuffle=True)

        trainloaders.append(trainloader)

    testloader = DataLoader(testset, batch_size=32)

    return trainloaders, testloader