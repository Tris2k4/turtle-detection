import os
import numpy as np
import cv2
from PIL import Image

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo

from loss import ValidationLoss

def get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes):
    cfg = _get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("val",)
    cfg.DATASETS.TEST = ()
    if device in ['cpu']:
        cfg.MODEL.DEVICE = 'cpu'
    else:
        cfg.MODEL.DEVICE = 'cuda'

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    cfg.SOLVER.BASE_LR = learning_rate

    cfg.SOLVER.MAX_ITER = iterations

    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes

    cfg.OUTPUT_DIR = output_dir
    return cfg


def get_dicts(img_dir, ann_dir):
	dataset_dicts = []

	for filename in os.listdir(img_dir):
		if not filename.endswith(('.jpg', '.jpeg', '.png')):
			continue

		record = {}

		# Image information
		image_path = os.path.join(img_dir, filename)
		img = cv2.imread(image_path)
		height, width = img.shape[:2]

		record["file_name"] = image_path
		record["image_id"] = filename
		record["height"] = height
		record["width"] = width

		# Annotation information
		ann_filename = os.path.splitext(filename)[0] + '.txt'
		ann_path = os.path.join(ann_dir, ann_filename)

		objs = []
		if os.path.exists(ann_path):
			with open(ann_path, 'r') as f:
				for line in f:
					class_id, *coords = line.strip().split()
					class_id = int(class_id)
					coords = [float(c) for c in coords]

					# Convert YOLO format to pixel coordinates
					num_points = len(coords) // 2
					px = [int(x * width) for x in coords[0::2]]
					py = [int(y * height) for y in coords[1::2]]

					poly = [(x, y) for x, y in zip(px, py)]

					obj = {
						"bbox": [min(px), min(py), max(px), max(py)],
						"bbox_mode": BoxMode.XYXY_ABS,
						"segmentation": [poly],
						"category_id": class_id,
					}
					objs.append(obj)

		record["annotations"] = objs
		dataset_dicts.append(record)

	return dataset_dicts


def register_datasets(root_dir, class_list_file):
    with open(class_list_file, 'r') as reader:
        classes_ = [l[:-1] for l in reader.readlines()]
    for d in ['train', 'val']:
        DatasetCatalog.register(d, lambda d=d: get_dicts(os.path.join(root_dir, d, 'imgs'),
                                                         os.path.join(root_dir, d, 'anns')))
        MetadataCatalog.get(d).set(thing_classes=classes_)

    print(classes_)
    return len(classes_)


def train(output_dir, data_dir, class_list_file, learning_rate, batch_size, iterations, checkpoint_period, device, model):
    nmr_classes = register_datasets(data_dir, class_list_file)
    cfg = get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()