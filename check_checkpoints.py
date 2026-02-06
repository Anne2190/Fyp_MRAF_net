import torch
from pathlib import Path

checkpoint_dirs = [
    'experiments/mraf_net_20260124_130245/checkpoints/best_model.pth',
    'experiments/mraf_net_20260203_180244/checkpoints/best_model.pth',
    'experiments/mraf_net_20260204_014915/checkpoints/best_model.pth'
]

print("=" * 80)
print("CHECKPOINT PERFORMANCE COMPARISON")
print("=" * 80)

for ckpt_path in checkpoint_dirs:
    path = Path(ckpt_path)
    if path.exists():
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            epoch = checkpoint.get('epoch', 'N/A')
            metrics = checkpoint.get('metrics', {})
            
            print(f"\n{path.parent.parent.name}")
            print(f"  Path: {ckpt_path}")
            print(f"  Epoch: {epoch}")
            print(f"  Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        except Exception as e:
            print(f"\n{ckpt_path}")
            print(f"  Error: {e}")
    else:
        print(f"\n{ckpt_path}")
        print(f"  Status: NOT FOUND")

print("\n" + "=" * 80)
