import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.io import read_image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

device_config = config.get('device', 'auto')
device = torch.device('cuda' if torch.cuda.is_available() and device_config == 'auto' else device_config)
logging.info(f'Using device: {device}')

try: 
    
    # Load both models concurrently and move to GPU if available
    with torch.no_grad():
        kweights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        mweights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=kweights).eval().to(device)
        mmodel = maskrcnn_resnet50_fpn(weights=mweights).eval().to(device)

    # Load and preprocess image
    image_path = config['image_path']
    try:
        image = read_image(image_path).to(device)
    except FileNotFoundError:
        print(f"Error: File {image_path} not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the image: {e}")
        exit()

    try:
        transforms = kweights.transforms()
        image_transformed = transforms(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"An error occurred during image transformation: {e}")
        exit()

    # Inference with both models
    try:
        with torch.no_grad():
            outputs, moutputs = model(image_transformed), mmodel(image_transformed)
    except Exception as e:
        print(f"An error occurred during model inference: {e}")
        exit()

    # Visualization
    fig, ax = plt.subplots()
    try:
        # Convert image tensor to NumPy for visualization, avoid conversion to PIL
        image_np = image.cpu().numpy().transpose((1, 2, 0))  # Change dimension order for matplotlib
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0,1] for correct display
        ax.imshow(image_np)
    except Exception as e:
        print(f"An error occurred during image visualization: {e}")
        exit()

    score_threshold = config.get('score_threshold', 0.75)
    class_colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5)]  # Red, Green, Blue with 50% opacity


    # Optimized Overlay Creation
    overlay = np.zeros((image_np.shape[0], image_np.shape[1], 4))  # Initialize an empty overlay
    for moutput in moutputs:
        masks = moutput['masks']
        scores = moutput['scores']
        labels = moutput['labels']
        for mask, score, label in zip(masks, scores, labels):
            if score > score_threshold:
                mask_np = mask.squeeze().cpu().numpy()
                color = class_colors[label % len(class_colors)]  # Cycle through colors for simplicity
                # Vectorized operation to apply mask to overlay
                overlay[mask_np > 0.5] = color

    ax.imshow(overlay, interpolation='nearest')  # Display the combined overlay

    # Bounding box and keypoint visualization
    for output in outputs:
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                rect = Rectangle(xy=(box[0].cpu().item(), box[1].cpu().item()), width=box[2].cpu().item() - box[0].cpu().item(), height=box[3].cpu().item() - box[1].cpu().item(),
                                color='r', fill=False)
                ax.add_patch(rect)
                ax.text(box[0].cpu().item(), box[1].cpu().item(), f'person: {score:.2f}', color='yellow', alpha=0.5)

        keypoints = output['keypoints'][output['scores'] > score_threshold]
        for idx, keypoint in enumerate(keypoints):
            for kpt in keypoint:
                x, y, v = kpt
                if v > 2.0:
                    ax.plot(x.cpu().item(), y.cpu().item(), 'go')
                    for start_p, end_p in [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
                        if keypoint[start_p][2] > 2.0 and keypoint[end_p][2] > 2.0:
                            ax.plot([keypoint[start_p][0].cpu().item(), keypoint[end_p][0].cpu().item()], [keypoint[start_p][1].cpu().item(), keypoint[end_p][1].cpu().item()], 'r-')

    ax.axis('off')
    plt.show()

except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")