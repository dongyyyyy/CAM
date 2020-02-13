# CAM ( Class Activation Map )

## resnet-CAN을 실행하면 테스트 가능 
### - 현재 input data 크기 128X128기준으로 10 epoch을 학습한 weight파일을 포함하고 있는 상태 (CIFAR10 dataset기준)
### - dataset의 경우에는 직접 하고 싶은 dataset에 맞춰서 pre-trained된 weight 파일을 사용하시면 됩니다. 해당 resnet18 model은 pytorch에서 제공하는 resnet과 동일한 모델 명으로 작성되어져 있기 때문에 같은 방식으로 활용하시면 됩니다.
### - 직접 학습을 하고 싶은 경우에는 dataset 이미지의 크기와 모델의 크기를 맞춰주시고 (현재 128X128로 되어있는 상태입니다. 빠른 학습을 위해) batch_size 및 start_epoch, is_train, classes_num을 본인의 상황에 맞게 수정한 후 사용하시면 됩니다.

