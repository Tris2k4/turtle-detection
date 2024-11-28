import torch
import yaml
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
from detectron2.config import get_cfg as detectron2_get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from util import get_cfg as util_get_cfg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_model(weights_path):
    """Load the trained Detectron2 model."""
    cfg = detectron2_get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Set the number of classes
    cfg.MODEL.WEIGHTS = weights_path  # Path to your custom trained weights
    cfg.MODEL.DEVICE = "cuda"  # Set the device to use (cuda or cpu)
    predictor = DefaultPredictor(cfg)
    return predictor


def load_class_names(data_yaml_path):
    """Load class names from the data.yaml file."""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']


def calculate_iou(pred_mask, true_mask):
    """Calculate IoU between prediction and ground truth masks."""
    # If both masks are empty (all False), return 1.0
    if not pred_mask.any() and not true_mask.any():
        return 1.0

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0
    return intersection / union


def visualize_masks(image, pred_masks, true_masks, class_names, image_name, save_dir="mask_visualizations"):
    """Visualize and save prediction and ground truth masks."""
    os.makedirs(save_dir, exist_ok=True)

    # Convert image to numpy array if it's a PIL image
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Define colors for each class (BGR format)
    colors = [
        (0, 255, 0),    # Green for class 0 (turtle)
        (0, 0, 255),    # Red for class 1 (flipper)
        (255, 0, 0)     # Blue for class 2 (head)
    ]

    # Create visualization for predictions
    pred_vis = img.copy()
    for cls_idx, mask in pred_masks.items():
        colored_mask = np.zeros_like(img)
        colored_mask[mask] = colors[cls_idx]
        pred_vis = cv2.addWeighted(pred_vis, 1, colored_mask, 0.5, 0)

        # Add class label
        cv2.putText(pred_vis, f"Class {cls_idx}: {class_names[cls_idx]}",
                    (10, 30 + cls_idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, colors[cls_idx], 2)

    # Create visualization for ground truth
    true_vis = img.copy()
    for cls_idx, mask in true_masks.items():
        colored_mask = np.zeros_like(img)
        colored_mask[mask] = colors[cls_idx]
        true_vis = cv2.addWeighted(true_vis, 1, colored_mask, 0.5, 0)

        # Add class label
        cv2.putText(true_vis, f"Class {cls_idx}: {class_names[cls_idx]}",
                    (10, 30 + cls_idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, colors[cls_idx], 2)

    # Combine visualizations horizontally
    combined = np.hstack((img, pred_vis, true_vis))

    # Add titles
    title_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, "Original", (img.shape[1]//2 - 70, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(title_bar, "Predictions", (img.shape[1] + img.shape[1]//2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(title_bar, "Ground Truth", (2*img.shape[1] + img.shape[1]//2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    final_vis = np.vstack((title_bar, combined))

    # Save the visualization
    cv2.imwrite(os.path.join(
        save_dir, f"{os.path.splitext(image_name)[0]}_masks.jpg"), final_vis)


def calculate_miou_per_image(model, test_dir, class_names):
    """Calculate mIoU for each class per image and visualize masks."""
    images_dir = os.path.join(test_dir, 'imgs')
    labels_dir = os.path.join(test_dir, 'anns')

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_results = {}

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        image_np = np.array(image)  # Convert PIL image to NumPy array

        class_ious = {i: 0.0 for i in range(len(class_names))}
        outputs = model(image_np)  # Pass NumPy array to the model

        img_height, img_width = image_np.shape[:2]

        class_pred_masks = {i: np.zeros((img_height, img_width), dtype=bool)
                            for i in range(len(class_names))}
        class_true_masks = {i: np.zeros((img_height, img_width), dtype=bool)
                            for i in range(len(class_names))}

        instances = outputs["instances"]
        pred_masks = instances.pred_masks.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()

        for mask, cls_idx in zip(pred_masks, pred_classes):
            mask = mask.astype(np.uint8)  # Convert mask to uint8
            mask = cv2.resize(mask, (img_width, img_height))
            mask = mask > 0.5
            class_pred_masks[int(cls_idx)] = np.logical_or(
                class_pred_masks[int(cls_idx)], mask)

        label_file = os.path.join(
            labels_dir, os.path.splitext(image_file)[0] + '.txt')
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) > 0:
                        cls_idx = int(data[0])
                        vertices = np.array([float(x)
                                            for x in data[1:]]).reshape(-1, 2)
                        vertices[:, 0] *= img_width
                        vertices[:, 1] *= img_height
                        vertices = vertices.astype(np.int32)

                        mask = np.zeros(
                            (img_height, img_width), dtype=np.uint8)
                        cv2.fillPoly(mask, [vertices], 1)
                        mask = mask.astype(bool)
                        class_true_masks[cls_idx] = np.logical_or(
                            class_true_masks[cls_idx], mask)

            # Calculate IoU for each class
            for cls_idx in range(len(class_names)):
                iou = calculate_iou(
                    class_pred_masks[cls_idx], class_true_masks[cls_idx])
                class_ious[cls_idx] = iou

        # Visualize masks
        visualize_masks(image, class_pred_masks, class_true_masks,
                        class_names, image_file)

        # Save individual image results
        save_image_miou(image_file, class_ious)

        # Store results
        image_results[image_file] = class_ious

    return image_results


def save_image_miou(image_name, class_ious, miou_dir="mIOU"):
    """Save mIoU results for a single image."""
    os.makedirs(miou_dir, exist_ok=True)
    output_file = os.path.join(
        miou_dir, f"{os.path.splitext(image_name)[0]}_miou.txt")

    with open(output_file, 'w') as f:
        for cls_idx, iou in class_ious.items():
            f.write(f"{cls_idx}: {iou:.4f}\n")


def calculate_final_miou(miou_dir="mIOU"):
    """Calculate mean IoU for each class from all result files."""
    class_values = {}

    # Read all files in the mIOU directory
    for filename in os.listdir(miou_dir):
        if filename.endswith('_miou.txt'):
            filepath = os.path.join(miou_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    cls_idx, iou_value = line.strip().split(':')
                    cls_idx = int(cls_idx)
                    iou_value = float(iou_value)

                    if cls_idx not in class_values:
                        class_values[cls_idx] = []
                    class_values[cls_idx].append(iou_value)

    # Calculate mean for each class
    final_results = {}
    for cls_idx, values in class_values.items():
        final_results[cls_idx] = np.mean(values)

    return final_results


def calculate_confusion_matrix(image_results, class_names):
    """Calculate confusion matrix for each class."""
    y_true = []
    y_pred = []

    for image_name, class_ious in image_results.items():
        for cls_idx, iou in class_ious.items():
            y_true.append(cls_idx)
            y_pred.append(1 if iou > 0.5 else 0)  # Assuming IoU > 0.5 as positive prediction

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    return cm


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def main():
    weights_path = "./output/model_0007999.pth"
    data_yaml_path = "yolo-dataset/data.yaml"
    test_dir = "yolo-dataset/test"

    model = load_model(weights_path)
    class_names = load_class_names(data_yaml_path)

    # Calculate mIoU per image
    image_results = calculate_miou_per_image(model, test_dir, class_names)

    # Save individual results
    for image_name, class_ious in image_results.items():
        save_image_miou(image_name, class_ious)

    # Calculate final mean IoU
    final_results = calculate_final_miou()

    # Save final results
    with open("final_miou_results.txt", 'w') as f:
        for cls_idx, mean_iou in final_results.items():
            f.write(f"{cls_idx}: {mean_iou}\n")

    # Calculate and plot confusion matrix
    cm = calculate_confusion_matrix(image_results, class_names)
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()
