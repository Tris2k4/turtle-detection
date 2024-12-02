### `calculate_precision.py`
This file calculates precision and recall metrics from the model's output metrics. It includes functions to:
- Calculate precision and recall from metrics data.
- Print precision and recall at different iterations and calculate overall metrics.

### `util.py`
This file contains utility functions for configuring and training the Detectron2 model. It includes:
- A function to get the configuration for the model.
- Functions to register datasets and train the model.

### `predict.py`
This file performs inference using a trained Detectron2 model. It includes:
- Functions to load class names and model configuration.
- Code to perform inference on test images and visualize the results.

### `loss.py`
- Calculate the loss of the model.

### `plotloss.py`
This file plots the training and validation loss over iterations. It includes:
- Functions to read metrics from a file and calculate moving averages.
- Code to plot the loss curves using Matplotlib.

### `calculate-segmentation-miou.py`
This file calculates the mean Intersection over Union (mIoU) for segmentation tasks. It includes:
- Functions to load the model and class names.
- Functions to calculate IoU, visualize masks, and save mIoU results.
- Code to calculate and plot the confusion matrix.

### `yolo_to_coco.py`
This file is to convert the data from yolo format to coco format

### `train_dectectron.py`
This file is use to train the model

### `train.ipynb`
This file contains a python command to run the training

To get the training model, please head to this link https://drive.google.com/drive/folders/1BcPwVqSxbcQ9siCiGkRLTVsPk4jY4gbL and go to the output folder.
