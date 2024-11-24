import torch
import torchfile

model_name = "resnet50_places365.pth.tar"
model = torch.load(model_name,  map_location=torch.device('cpu'))
print(model)