import matplotlib.pyplot as plt
import numpy as np

def plot_losses(losses_cls, losses_box, save_path):
    plt.plot(losses_cls, label="Classification Loss")
    plt.plot(losses_box, label="Box Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(precisions, recalls, ap, save_dir):
    """Plot and save precision-recall curve with AP value."""
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', label=f'Precision-Recall (AP={ap:.3f})')
    plt.fill_between(recalls, precisions, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    
    save_path = save_dir / 'precision_recall_curve.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_map_progress(aps, save_dir):
    """Plot mAP progress over the test set."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(aps) + 1), aps, 'g-')
    plt.fill_between(np.arange(1, len(aps) + 1), aps, alpha=0.2)
    plt.xlabel('Images Processed')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision Progress')
    plt.grid(True)
    
    # Calculate and display final mAP
    final_map = np.mean(aps)
    plt.axhline(y=final_map, color='r', linestyle='--', 
                label=f'Final mAP: {final_map:.3f}')
    plt.legend()
    
    save_path = save_dir / 'map_progress.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_epoch_metrics(metric_dicts, save_path):
    epochs = list(range(1, len(metric_dicts) + 1))
    f1s = [m['F1 Score'] for m in metric_dicts]
    maps = [m['mAP'] for m in metric_dicts]
    avg_ious = [m['Average IoU'] for m in metric_dicts]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, f1s, label='F1 Score')
    plt.plot(epochs, maps, label='mAP')
    plt.plot(epochs, avg_ious, label='Average IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Epoch-wise Evaluation Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

def plot_iou_histogram(iou_list, save_dir):
    """Plot histogram of IoU values from matched detections."""
    plt.figure(figsize=(8, 6))
    plt.hist(iou_list, bins=20, range=(0, 1), color='skyblue', edgecolor='black')
    plt.title('Distribution of IoUs for Matched Detections')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    
    save_path = save_dir / 'iou_histogram.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
