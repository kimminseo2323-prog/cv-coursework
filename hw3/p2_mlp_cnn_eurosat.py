import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder, utils
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import os


os.chdir("E://김민서//대학원//컴퓨터비전//HW//HW3_김민서//")
print(os.getcwd())

#EuroSAT 데이터셋 다운로드 코드
#만약 다운로드 코드에 에러가 발생한다면, 아래 링크에서 직접 다운로드 후 압축을 해제해 주세요.
def get_EuroSAT(dirname):
    import os
    if(os.path.exists(dirname)):
        print("Dataset is already exist.")
        return(os.path.join(dirname, '2750'))
    
    os.makedirs(dirname, exist_ok=True)
    utils.download_and_extract_archive(
                "http://madm.dfki.de/files/sentinel/EuroSAT.zip",
                download_root=dirname,
                md5="c8fa014336c82ac7804f0398fcb19387",
                remove_finished=True,
            )
    return(os.path.join(dirname, '2750'))

# GPU 사용이 가능하다면 사용하고, 그렇지 않다면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''P-2.1'''

'''
data를 변형시키는 augmentation (rotation, color)을 추가하는 것은 가능합니다.
사용하신다면 리포트에 관련 내용및 사용한 이유를 적어주세요.
'''

# 데이터 전처리 및 로드
transform = transforms.Compose([
    #---------------------------------------------------#
    transforms.Resize((32, 32)),  # 모든 이미지를 32×32 크기로 통일
    #---------------------------------------------------#
   
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder(get_EuroSAT('EuroSAT'), transform=transform)

# 학습, 검증, 테스트 데이터 분할
dataset_size = len(dataset)
classes = dataset.classes
dataset_size = len(dataset) # Dataset size: 27000
indices = np.arange(dataset_size)
np.random.shuffle(indices)

#아래 split에서 데이터셋을 적절하게 train, val, test로 분할하세요.
#코드를 보고 None에 적절한 인수값을 넣어 dataset을 분리시키세요

# 60% / 20% / 20% split
split1 = int(0.6 * dataset_size)
split2 = int(0.8 * dataset_size)

np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)


print('Train loader size:', len(train_loader.dataset))  # Train loader size: 16200
print('Val loader size:', len(val_loader.dataset))      # Val loader size: 5400
print('Test loader size:', len(test_loader.dataset))    # Test loader size: 5400



'''P-2.2'''
# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(MLP, self).__init__()
        # 입력: 3*32*32 → 은닉: 512 → 은닉: 256 → 출력: num_classes
        self.fc1 = nn.Linear(3*32*32, 512) # 입력 3072 → 은닉 512
        self.fc2 = nn.Linear(512, 256) # 은닉 512 → 은닉 256
        self.fc3 = nn.Linear(256, num_classes) # 은닉 256 → 출력 num_classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1) # (B,3,32,32) → (B,3072)
        x = self.relu(self.fc1(x)) # 은닉층1 + ReLU
        x = self.relu(self.fc2(x)) # 은닉층2 + ReLU
        x = self.fc3(x)  # 출력층
        return x

'''P-2.3'''
class CNN(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(CNN, self).__init__()
        # Conv 블록 1 : 3→32 채널, 3×3, padding=1, ReLU, MaxPool2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.act1  = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        # Conv 블록 2 : 32→64 채널, 3×3, padding=1, ReLU, MaxPool2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2  = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        # Fully-connected : 64×8×8 → 256 → num_classes
        self.fc1   = nn.Linear(64*8*8, 256)
        self.act3  = nn.ReLU()
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x))) # (B,3,32,32) → (B,32,16,16)
        x = self.pool2(self.act2(self.conv2(x))) # (B,64,8,8)
        x = x.view(x.size(0), -1) # flatten → (B,64*8*8)
        x = self.act3(self.fc1(x)) # FC 은닉 + ReLU
        x = self.fc2(x)
        return x

# 모델 초기화 및 손실 함수, 최적화 알고리즘 설정
model = CNN().to(device)  # 또는 MLP().to(device)
# loss objective : cross entropy
criterion = nn.CrossEntropyLoss()

'''P-2.4'''
import torch.optim as optim
#여기에 최적화 함수를 정의하세요. https://pytorch.org/docs/stable/optim.html 참고하세요.
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)

# 학습 함수
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss


# 검증 함수
def validate(model, loader, criterion):
    if loader is None:
        loader = test_loader
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = running_loss / len(loader.dataset)
    val_acc  = accuracy_score(all_labels, all_preds)
    cm       = confusion_matrix(all_labels, all_preds)
    return val_loss, val_acc, cm                             # ← 3개를 반환


# 학습 및 검증 실행
#전체 epoch은 100으로 설정
# epochs 50,100,150 테스트
num_epochs = 100

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy, _ = validate(model, val_loader, criterion)
    print(f"Iter {epoch + 1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}")
 
# train, validate 함수를 수정하여 accuracy와 confusion matrix를 계산하세요.
# 예시인 sklearn의 confusion matrix함수는 다음과 같이 구현할 수 있습니다.

_, test_acc, test_cm = validate(model, None, criterion)
print(f"Test Accuracy: {test_acc:.4f}")

# train 데이터에 대한 결과 출력
_, train_acc, train_cm = validate(model, train_loader, criterion)
print(f"Train Accuracy: {train_acc:.4f}")


# 이 confusion matrix의 시각화는 sklearn.metrics.ConfusionMatrixDisplay로 할 수도 있지만, seaborn을 사용한 예시는 다음과 같습니다.
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,8))
hmhm = sns.heatmap(
    test_cm,                     # ← 실제 2D array
    annot=True, fmt="d", cmap="Blues",
    xticklabels=classes,         # 클래스 이름 리스트
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Test Confusion Matrix (Acc: {test_acc:.4f})")

save_path = os.path.join('result', '2','test_confusion_matrix.png')
hmhm.figure.savefig(save_path)
plt.close()
