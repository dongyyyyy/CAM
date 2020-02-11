import torch, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


def trainloader_func(batch_size = 100):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    train_data = datasets.ImageFolder('./dataset/dog_cat/', transform=transform_train)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

def testloader_func(batch_size = 100):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    test_data = datasets.ImageFolder('./dataset/dog_cat/', transform=transform_test)
    return DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
