import torch.nn as nn
import torchvision

class ResNetCNN(nn.Module):
    def __init__(self):
        super(ResNetCNN, self).__init__()
        
        # ResNet18 모델 불러오기
        self.resnet = torchvision.models.resnet18(pretrained=True)
        # ResNet의 마지막 FC 레이어를 새로운 레이어로 대체
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)  # 출력 노드 개수는 이진 분류 문제에 맞게 2로 설정
        
    def forward(self, x):
        x = self.resnet(x)
        return x