import json
import numpy as np

def calculate_metrics(metrics_data):
    """
    Calculate precision and recall from metrics data
    """
    # Extract relevant metrics
    mask_accuracy = metrics_data.get('mask_rcnn/accuracy', 0)
    mask_false_negative = metrics_data.get('mask_rcnn/false_negative', 0) 
    mask_false_positive = metrics_data.get('mask_rcnn/false_positive', 0)
    
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    # TP = correctly predicted positive cases
    # Using accuracy as an approximation of true positive rate
    TP = mask_accuracy
    
    # FP and FN are directly provided
    FP = mask_false_positive
    FN = mask_false_negative
    
    # Calculate precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Calculate recall: TP / (TP + FN)  
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return precision, recall

def print_metrics_over_time(metrics_file):
    """
    Print precision and recall at different iterations and calculate overall metrics
    """
    all_precisions = []
    all_recalls = []
    
    with open(metrics_file) as f:
        for line in f:
            metrics = json.loads(line)
            precision, recall = calculate_metrics(metrics)
            all_precisions.append(precision)
            all_recalls.append(recall)
            
            iteration = metrics.get('iteration')
            print(f"Iteration {iteration}:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print("-" * 30)
    
    # Calculate overall metrics
    overall_precision = np.mean(all_precisions)
    overall_recall = np.mean(all_recalls)
    
    print("\nOverall Metrics:")
    print(f"Average Precision: {overall_precision:.4f}")
    print(f"Average Recall: {overall_recall:.4f}")
    print(f"F1 Score: {2 * (overall_precision * overall_recall) / (overall_precision + overall_recall):.4f}")

# Use the function
print_metrics_over_time('./output/metrics.json')