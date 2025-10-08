import torch
import numpy as np
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, device, tokenize_fn):
    model.train()
    total_loss = 0
    
    for imgs, texts, labels in tqdm(train_loader):
        imgs = imgs.to(device)
        text_inputs = tokenize_fn(texts).to(device)
        
        logits = model(imgs, text_inputs)
        
        labels = torch.arange(len(imgs)).to(device)
        loss_i2t = torch.nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = torch.nn.CrossEntropyLoss()(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def zero_shot_classification(model, image, class_descriptions, tokenize_fn, device):
    model.eval()
    with torch.no_grad():
        image_feature = model.image_encoder(image.unsqueeze(0).to(device))
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        
        text_inputs = tokenize_fn(class_descriptions).to(device)
        text_features = model.text_encoder(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = model.logit_scale.exp()
        logits = (image_feature @ text_features.t()) * logit_scale
        probs = logits.softmax(dim=-1)
    
    return probs.cpu().numpy()
