# CAM ( Class Activation Map )

## Pytorch로 작성되어 있습니다. 사용하시기 위해서는 해당 라이브러리를 설치하셔야 합니다.
### - opencv
### - pytorch
### - torchsummary(해당 패키지를 import하는 부분이 있어서 해당 부분을 지우고하실 경우에는 필요하지는 않습니다.)

---

## resnet-CAN을 실행하면 테스트 가능 
#### - 현재 input data 크기 128X128기준으로 10 epoch을 학습한 weight파일을 포함하고 있는 상태 (CIFAR10 dataset기준)
#### - dataset의 경우에는 직접 하고 싶은 dataset에 맞춰서 pre-trained된 weight 파일을 사용하시면 됩니다. 해당 resnet18 model은 pytorch에서 제공하는 resnet과 동일한 모델 명으로 작성되어져 있기 때문에 같은 방식으로 활용하시면 됩니다.
#### - 직접 학습을 하고 싶은 경우에는 dataset 이미지의 크기와 모델의 크기를 맞춰주시고 (현재 128X128로 되어있는 상태입니다. 빠른 학습을 위해) batch_size 및 start_epoch, is_train, classes_num을 본인의 상황에 맞게 수정한 후 사용하시면 됩니다.
#### - 해당 소스와 같이 resnet 클래스를 따로 만들 필요 없이 위에서 언급했듯이 pytorch에서 제공하는 resnet모델을 활용하셔도 전혀 상관 없습니다.
#### - 논문 관련된 정리 내용을 보시고 싶으신 경우에는 [티스토리 블로그](https://dydeeplearning.tistory.com/9)에 내용을 봐주시면 조금은 도움이 되실 수도 있습니다. 실제 논문은 [여기](https://arxiv.org/abs/1512.04150)를 클릭하시면 보실 수 있습니다.

#### - 추가적으로 정리할 예정입니다.

