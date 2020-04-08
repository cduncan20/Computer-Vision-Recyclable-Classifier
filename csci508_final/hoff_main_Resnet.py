import torch
from torchvision import datasets, models, transforms
from PIL import Image

def main():
    print("Loading in pretrained network ...")
    model = models.resnet18(pretrained=True)    # Get pretrained network

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    img = Image.open("data/dog.jpg")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    model.eval()
    out = model(batch_t)
    print("Output scores shape: ", out.shape)

    with open('data/imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(labels[index[0]], percentage[index[0]].item())

    _, indices = torch.sort(out, descending=True)
    print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])

if __name__ == "__main__":
    main()
