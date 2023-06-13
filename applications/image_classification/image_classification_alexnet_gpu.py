import configs
from utils import current_milli_time
from configs import *
from protos import benchmark_pb2 as pb2
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Check if a GPU is available and set the device accordingly
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("[x] Device found: {0}".format(device))
#
# if configs.EDGE_DEVICE_NAME == EdgeDevice.NANO:
#     alexnet = models.alexnet(pretrained=True)
# else:
#     alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
# torch.save(alexnet.state_dict(), 'alexnet.pth')

def classify_image(request, request_received_time_ms):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = models.alexnet(num_classes=1000).to(device)
    model.transform = transform
    model.load_state_dict(torch.load('alexnet.pth'))
    model.eval()  # Set the model to evaluation mode

    image = request.image.to(device)
    output = model(image.unsqueeze(0))
    prob, predicted = torch.max(output, 1)

    classification_response = pb2.ImageClassificationResponse()
    classification_response.top_category_id = int(predicted.item())
    classification_response.top_category_probability = int(prob.item() * 100)
    classification_response.request_time_ms = request.request_time_ms
    classification_response.request_received_time_ms = request_received_time_ms
    classification_response.response_time_ms = current_milli_time()

    return classification_response
