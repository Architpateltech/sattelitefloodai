from torchvision import datasets, transforms

path = "../datasets/flood/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(path, transform=transform)

print("Classes:", dataset.classes)
print("Total Images:", len(dataset))