"""
MRAF-Net Visualization Script
Plot training curves and confusion matrix

Author: Anne Nidhusha Nithiyalan (w1985740)
"""

import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tb_logs(log_dir: str):
    """Extract scalars from TensorBoard logs."""
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    data = {}
    
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_time': [e.wall_time for e in events]
        }
        
    return data


def plot_curves(data: dict, output_dir: Path):
    """Plot training and validation curves."""
    sns.set_theme(style="whitegrid")
    
    # Loss Plot
    if 'train/loss' in data or 'val/val_loss' in data:
        plt.figure(figsize=(10, 6))
        if 'train/loss' in data:
            plt.plot(data['train/loss']['steps'], data['train/loss']['values'], label='Train Loss')
        if 'val/val_loss' in data:
            plt.plot(data['val/val_loss']['steps'], data['val/val_loss']['values'], label='Val Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(output_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Dice Metrics Plot
    dice_tags = [t for t in data.keys() if t.startswith('val/dice_')]
    if dice_tags:
        plt.figure(figsize=(10, 6))
        for tag in dice_tags:
            label = tag.replace('val/dice_', '').upper()
            plt.plot(data[tag]['steps'], data[tag]['values'], label=f'Dice {label}')
        plt.xlabel('Steps')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Scores')
        plt.legend()
        plt.savefig(output_dir / 'dice_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_confusion_matrix(cm_data: list, output_path: Path):
    """Plot confusion matrix heatmap."""
    cm = np.array(cm_data)
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle div by zero
    
    classes = ['Background', 'NCR/NET', 'ED', 'ET']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Voxel-wise Confusion Matrix (Normalized)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MRAF-Net Visualization tool')
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to experiment directory')
    parser.add_argument('--eval_json', type=str, default=None, help='Path to evaluation results JSON')
    
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    vis_dir = exp_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 1. Process Training curves
    log_dir = exp_dir / 'logs'
    if log_dir.exists():
        print(f"Extracting logs from {log_dir}...")
        try:
            data = extract_tb_logs(log_dir)
            if data:
                plot_curves(data, vis_dir)
                print(f"Curves saved to {vis_dir}")
            else:
                print("No data found in TensorBoard logs.")
        except Exception as e:
            print(f"Error extracting logs: {e}")
    
    # 2. Process Confusion Matrix
    eval_json_path = args.eval_json
    if eval_json_path is None:
        # Try default location: outputs/evaluation/evaluation_results.json or inside exp_dir
        eval_json_path = exp_dir / 'evaluation_results.json'
        if not eval_json_path.exists():
            eval_json_path = Path('outputs/evaluation/evaluation_results.json')

    if Path(eval_json_path).exists():
        print(f"Loading evaluation results from {eval_json_path}...")
        with open(eval_json_path, 'r') as f:
            results = json.load(f)
        
        if 'aggregate' in results and 'confusion_matrix' in results['aggregate']:
            cm_data = results['aggregate']['confusion_matrix']
            plot_confusion_matrix(cm_data, vis_dir / 'confusion_matrix.png')
            print(f"Confusion matrix saved to {vis_dir / 'confusion_matrix.png'}")
        else:
            print("Confusion matrix data not found in JSON.")
    else:
        print(f"Evaluation results not found at {eval_json_path}. Run evaluation first.")


if __name__ == '__main__':
    main()
