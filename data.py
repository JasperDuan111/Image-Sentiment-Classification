import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob

class EmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 情感类别映射
        self.emotion_dict = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5,
            'surprise': 6
        }
        
        self.image_paths = []
        self.labels = []
        
        for emotion, label in self.emotion_dict.items():
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                # 获取该情感类别下的所有图片
                image_files = glob.glob(os.path.join(emotion_dir, '*.jpg'))
                self.image_paths.extend(image_files)
                self.labels.extend([label] * len(image_files))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label
    
    def get_class_names(self):
        return list(self.emotion_dict.keys())

def get_data(batch_size=32, train_dir='data/train', test_dir='data/test'):
    train_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomRotation(degrees=10), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    train_dataset = EmotionDataset(train_dir, transform=train_transforms)

    test_dataset = EmotionDataset(test_dir, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    print(f"Train Set: {len(train_dataset)}")
    print(f"Test Set: {len(test_dataset)}")
    print(f"Batch Size: {batch_size}")
    print(f"Train Batch: {len(train_loader)}")
    print(f"Test Batch: {len(test_loader)}")
    print(f"Sentiment Class: {train_dataset.get_class_names()}")
    
    return train_loader, test_loader