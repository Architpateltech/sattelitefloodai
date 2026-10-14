import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim



DATASET_PATH = "../datasets/flood/train"



transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]
)



dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)



loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)



print(
    "Classes:",
    dataset.classes
)



# Pretrained ResNet

model = models.resnet18(
    weights="DEFAULT"
)



# Replace final layer

model.fc = nn.Linear(
    model.fc.in_features,
    2
)



device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



model.to(device)



criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)



epochs = 10



for epoch in range(epochs):

    total_loss = 0


    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        output = model(images)


        loss = criterion(
            output,
            labels
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()



    print(
        f"Epoch {epoch+1}/{epochs}",
        "Loss:",
        total_loss
    )



torch.save(
    model.state_dict(),
    "flood_model.pth"
)


print("Model Saved")