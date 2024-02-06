import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.io import read_image
from torchvision.utils import draw_keypoints
import torchvision.transforms.functional as F  # Import statement added
import matplotlib.pyplot as plt
from pathlib import Path

weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
model = keypointrcnn_resnet50_fpn(weights=weights).eval()

# Load an image
image_path = str(Path("./test_1.jpg"))  # Update this path
image = read_image(image_path)

# Assuming you have a function for transformations or use the default ones
transforms = weights.transforms()
image_transformed = transforms(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(image_transformed)
    
score_threshold = 0.75
for output in outputs:
    keypoints = output['keypoints']
    scores = output['scores']
    high_confidence_idxs = scores > score_threshold
    keypoints = keypoints[high_confidence_idxs]

    # Draw keypoints on the original image
    result_image = draw_keypoints(image, keypoints.squeeze(1), colors="blue", radius=2)

    # Convert tensor to PIL Image for plotting
    result_image_pil = F.to_pil_image(result_image)
    plt.imshow(result_image_pil)
    plt.axis('off')
    plt.show()