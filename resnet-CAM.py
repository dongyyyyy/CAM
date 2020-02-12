from resnet_train import *
from dataset.CIFAR10 import *
import torch.nn as nn
from PIL import Image
from CAM import *
from torch.optim import lr_scheduler
from models.resnet18 import *
import os
from timeit import default_timer as timer

classes_num = 10 # dog , cat

batch_size = 500 # 이미지 크기 128X128기준 rtx 2080ti로 학습할 수 있는 큰 batch_size 그래픽 메모리를 약 9.2GB 사용
#batch_size = 150 # 이미지 크기 224X224기준 rtx 2080ti로 학습할 수 있는 큰 batch_Size
#learning_rate = 0.0001
learning_rate = 0.001
PRETRAINED = False

classes = {0:'plane', 1:'car', 2:'bird', 3:'cat',
           4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}


if __name__ == "__main__":
    os.makedirs("checkpoint",exist_ok=True)
    os.makedirs("result", exist_ok=True)
    os.makedirs("checkCAM",exist_ok=True)
    start_epoch = 10
    is_train = False
    print('start!')

    #resnet18 = torch.hub.load('pytorch/vision:v0.5.0','resnet18',pretrained=True)

    trainloader, testloader= CIFAR10(batch_size=batch_size)
    #trainloader = trainloader_func(batch_size=100)
    #testloader = testloader_func(batch_size=100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet18_custom = resnet18(
        pretrained=PRETRAINED)  # 클래스 전체에 대한 pretrained 모델 parameter를 넣어주기 때문에 우선적으로 class 1000 기준으로 모델을 만듬

    in_features = resnet18_custom.fc.in_features  # fc에서의 (classification) input_filter의 크기를 가져온다
    #resnet18_custom.fc = nn.Linear(in_features, classes_num)  # 해당 fc의 구성을 input = 512 output = 2로 수정
    #resnet18_custom = resnet18_custom.to(device)  #
    #summary(resnet18_custom, (3, 224, 224))

    if PRETRAINED == True:
        for param in resnet18_custom.parameters():
            param.requires_grad = False
        resnet18_custom.fc = nn.Linear(in_features, classes_num)  # 해당 fc의 구성을 input = 512 output = 2로 수정
    else:
        resnet18_custom.fc = nn.Linear(in_features, classes_num)  # 해당 fc의 구성을 input = 512 output = 2로 수정
    final_layer = 'layer4' # GAP 직전의 feature map을 가져오기 위한 위치
    resnet18_custom = resnet18_custom.to(device)  #

    if start_epoch != 0:
        print("Load weight file : checkpoint/%d.pt"%start_epoch)
        resnet18_custom.load_state_dict(torch.load("checkpoint/%d.pt"%start_epoch))

    criterion = torch.nn.CrossEntropyLoss()

    if PRETRAINED:
        optimizer = torch.optim.Adam(resnet18_custom.fc.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(resnet18_custom.fc.parameters(),lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(resnet18_custom.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(resnet18_custom.parameters(),lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # 7 에폭마다 0.1씩 학습율 감소
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if is_train:
        for epoch in range(start_epoch+1,start_epoch+11): # 한번 학습 수행시마다 10번의 epoch하도록 설정
            #for params_group in optimizer.param_groups:
            #    print("lr: ", params_group['lr'])
            start_time = timer()
            fine_tuning(trainloader=trainloader,model=resnet18_custom,epoch = epoch, criterion = criterion, optimizer = optimizer,scheduler= exp_lr_scheduler)
            print("Total Time : ",timer() - start_time)
            test(testloader=testloader,model=resnet18_custom,epoch = epoch, criterion = criterion)
    else:
        test(testloader=testloader, model=resnet18_custom, epoch=start_epoch, criterion=criterion)
    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        print(output)
        features_blobs.append(output.data.cpu().numpy())

    resnet18_custom._modules.get(final_layer).register_forward_hook(hook_feature)


    root = 'sample.jpg'
    img = Image.open(root)
    get_cam_CIFAR10(resnet18_custom, features_blobs, classes)
    #get_cam(resnet18_custom, features_blobs, img, classes, root)

