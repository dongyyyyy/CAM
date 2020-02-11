import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
        mean=mean,
        std=std
    )

CIFAR_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

class UnNormalize(object):
    def __init__(self):
        self.mean = mean
        self.std = std
    def __call__(self,tensor):
        for t,m,s in zip(tensor,self.mean,self.std):
            t.mul_(s).add_(m)
        return tensor

def CIFAR10(batch_size = 100):
    trainset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10/', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    # functions to show an image
    return trainloader,testloader


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # get some random training images
    trainloader, testloader = CIFAR10()
    dataiter = iter(trainloader)

    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    unorm = UnNormalize()
    images  = unorm(images)
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % CIFAR_classes[labels[j]] for j in range(4)))

