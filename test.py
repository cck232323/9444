import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_labels = []
        
        with open(txt_file, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split(' ')
                self.image_labels.append((parts[0], int(parts[1])))
                
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('L')  # 转换为灰度图像
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def main():
    # 创建TensorBoard写入器
    writer = SummaryWriter('runs/experiment_name')
    
    # 加载数据集
    train_dataset = CustomDataset(txt_file='/Users/kim/Downloads/hw2/ip102_v1.1/train.txt', root_dir='/Users/kim/Downloads/ip102_v1.1/images', transform=transform)
    val_dataset = CustomDataset(txt_file='/Users/kim/Downloads/hw2/ip102_v1.1/val.txt', root_dir='/Users/kim/Downloads/ip102_v1.1/images', transform=transform)
    test_dataset = CustomDataset(txt_file='/Users/kim/Downloads/hw2/ip102_v1.1/test.txt', root_dir='/Users/kim/Downloads/ip102_v1.1/images', transform=transform)

    # 使用较小的训练集
    small_train_dataset = Subset(train_dataset, range(4500))
    # 使用完整的验证集和测试集
    small_val_dataset = Subset(val_dataset, range(700))
    small_test_dataset = test_dataset

    train_loader = DataLoader(small_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(small_val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(small_test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 定义MobileNetV2模型
    class MobileNetV2(nn.Module):
        def __init__(self, num_classes=102):
            super(MobileNetV2, self).__init__()
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 修改第一层以接受灰度图像
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
        def forward(self, x):
            return self.model(x)

    model = MobileNetV2(num_classes=102).to(device)

    # 稀疏正则化
    def sparse_regularization(model, lambda_):
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                reg_loss += lambda_ * torch.sum(torch.abs(param))
        return reg_loss

    # 训练和验证函数
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
        best_model_wts = model.state_dict()
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
        
            # 每个epoch有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    dataloader = train_loader
                else:
                    model.eval()
                    dataloader = val_loader
                
                running_loss = 0.0
                running_corrects = 0
            
                # 遍历数据
                for i, (inputs, labels) in enumerate(dataloader):
                    if inputs is None or labels is None:
                        continue
                
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                    optimizer.zero_grad()
                
                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                        # 添加稀疏正则化
                        if phase == 'train':
                            loss += sparse_regularization(model, lambda_=0.001)
                        
                        # 反向传播 + 优化
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                
                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
                
                    if i % 10 == 0:  # 每处理10批数据打印一次进度
                        print(f'Processed {i * len(inputs)} / {len(dataloader.dataset)} images')
                
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # 记录损失和准确率到TensorBoard
                writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase} accuracy', epoch_acc, epoch)
            
                # 深拷贝模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
    
        print(f'Best val Acc: {best_acc:4f}')
    
        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)
        return model

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

    # 测试模型
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # 存储每张图片的loss和准确度
    test_results = []

    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
            # 记录每张图片的loss和准确度
            for j in range(inputs.size(0)):
                result = {
                    'image': test_dataset.image_labels[i * test_loader.batch_size + j][0],
                    'loss': loss.item(),
                    'correct': preds[j].item() == labels[j].item()
                }
                test_results.append(result)
        
        if i % 10 == 0:  # 每处理10批数据打印一次进度
            print(f'Processed {i * len(inputs)} / {len(test_loader.dataset)} test images')

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}')

    # 打印每张图片的loss和准确度
    for result in test_results:
        print(f"Image: {result['image']}, Loss: {result['loss']:.4f}, Correct: {result['correct']}")

    # 关闭TensorBoard写入器
    writer.close()

if __name__ == '__main__':
    main()
