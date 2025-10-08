import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from models import CLIPModel
from utils.dataset import text_descriptions
from utils.tokenizer import tokenize_texts
from utils.helpers import zero_shot_classification
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = CLIPModel()
    model.load_state_dict(torch.load("outputs/checkpoints/final_model.pt", map_location=device))
    model.to(device)
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
    
    # 选择几个样本进行演示
    for i in range(5):
        test_image, true_label = test_dataset[i]
        class_descriptions = list(text_descriptions.values())
        
        probs = zero_shot_classification(model, test_image, class_descriptions, tokenize_texts, device)
        predicted_label = np.argmax(probs)
        
        print(f"样本 {i+1}:")
        print(f"  真实: {text_descriptions[true_label]}")
        print(f"  预测: {text_descriptions[predicted_label]}")
        print(f"  置信度: {probs[0][predicted_label]:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
