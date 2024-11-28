import os
import json
from PIL import Image

def yolo_to_coco(data_dir, class_names):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories
    for idx, class_name in enumerate(class_names):
        coco_format["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "turtle"
        })
    
    annotation_id = 0
    img_dir = os.path.join(data_dir, "imgs")
    ann_dir = os.path.join(data_dir, "anns")
    
    for img_id, img_name in enumerate(os.listdir(img_dir)):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        width, height = img.size
        
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(ann_dir, txt_name)
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id, x_center, y_center, w, h = map(float, line.strip().split())
                        
                        # Convert YOLO format to COCO format
                        x = (x_center - w/2) * width
                        y = (y_center - h/2) * height
                        w = w * width
                        h = h * height
                        
                        # Create a simple polygon for segmentation
                        segmentation = [[
                            x, y,
                            x + w, y,
                            x + w, y + h,
                            x, y + h
                        ]]
                        
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": int(class_id),
                            "bbox": [x, y, w, h],
                            "segmentation": segmentation,
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
    
    return coco_format