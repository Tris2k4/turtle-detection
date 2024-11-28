from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import numpy as np
import os

def load_class_names(class_names_path):
    """Load class names from the class.names file."""
    with open(class_names_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Load class names
class_names_path = "./class.names"
if not os.path.exists(class_names_path):
    print(f"Error: class.names file not found at {class_names_path}")
    exit(1)

class_names = load_class_names(class_names_path)

# Load config from a config file
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)  # Set the number of classes
cfg.MODEL.WEIGHTS = './output/model_0007999.pth'  # Path to your custom trained weights
cfg.MODEL.DEVICE = 'cuda'

# Create predictor instance
predictor = DefaultPredictor(cfg)

# Create result directory if it doesn't exist
result_dir = "./yolo-dataset/result/imgs/"
os.makedirs(result_dir, exist_ok=True)

# Directory containing test images
test_images_dir = "./yolo-dataset/test/imgs/"

# Loop over all images in test_images_dir
for image_name in os.listdir(test_images_dir):
    image_path = os.path.join(test_images_dir, image_name)
    # Check if image file
    if not image_name.endswith(('.jpg', '.jpeg', '.png')):
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        continue

    # Perform inference
    outputs = predictor(image)

    # Define a color map for each class
    color_map = {
        'carapace': (0, 255, 0),    # Green
        'flipper': (0, 0, 255),     # Red
        'head': (255, 0, 0)         # Blue
    }

    # Create a blank mask image
    mask_image = np.zeros(image.shape, dtype=np.uint8)

    # Process results
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()

    for i, (mask, class_id, score) in enumerate(zip(masks, classes, scores)):
        # Convert mask to uint8
        mask = (mask > 0.5).astype(np.uint8) * 255

        class_name = class_names[class_id]
        color = color_map.get(class_name, (255, 255, 255))  # Default to white if class not in color_map

        print(f"Detected {class_name} with confidence {score:.2f}")

        # Create a colored mask for this instance
        colored_mask = np.zeros(image.shape, dtype=np.uint8)
        colored_mask[mask > 0] = color

        # Add this mask to the overall mask image
        mask_image = cv2.addWeighted(mask_image, 1, colored_mask, 1, 0)

        # Find the centroid of the mask to place the label
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])

            # Put the class name at the centroid of the mask
            cv2.putText(image, class_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Blend the original image with the mask image
    result = cv2.addWeighted(image, 0.7, mask_image, 0.3, 0)

    # Save the result image
    result_image_path = os.path.join(result_dir, image_name)
    cv2.imwrite(result_image_path, result)

    print(f"Saved result image to {result_image_path}")