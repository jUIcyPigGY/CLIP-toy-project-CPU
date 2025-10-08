from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# 文本标签描述
text_descriptions = {
    0: "a t-shirt",
    1: "a pair of trousers",
    2: "a pullover",
    3: "a dress",
    4: "a coat",
    5: "a sandal",
    6: "a shirt",
    7: "a sneaker",
    8: "a bag",
    9: "an ankle boot"
}

class ClipDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        text = text_descriptions[label]
        return img, text, label

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
    
    train_loader = DataLoader(ClipDataset(train_dataset), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ClipDataset(test_dataset), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, text_descriptions
