import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from models import CLIPModel
from utils.dataset import get_dataloaders
from utils.tokenizer import tokenize_texts
from utils.helpers import train_epoch

def main():
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 5
    embed_dim = 128
    lr = 1e-4
    
    # 创建输出目录
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    # 获取数据
    train_loader, test_loader, text_descriptions = get_dataloaders(batch_size)
    
    # 初始化模型和优化器
    model = CLIPModel(embed_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device, tokenize_texts)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"outputs/checkpoints/checkpoint_epoch_{epoch+1}.pt")
    
    # 保存最终模型
    torch.save(model.state_dict(), "outputs/checkpoints/final_model.pt")
    print("训练完成，模型已保存!")

if __name__ == "__main__":
    main()
