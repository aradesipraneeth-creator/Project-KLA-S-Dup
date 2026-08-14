import os
import sys
import torch

from models.airnet import AIRNet
from utils.edge_utils import compute_sobel_edges
from losses.hybrid_loss import AIRNetHybridLoss

def main():
    print("====================================================")
    print("AIR-NET V1 FORWARD PASS & STABILITY TEST")
    print("====================================================")

    # 1. Instantiate AIRNet Model
    model = AIRNet(
        in_channels=1,
        out_channels=1,
        dim=32,
        channels=[32, 64, 128, 192],
        heads=[1, 2, 4, 6],
        enc_blocks=[2, 2, 4],
        latent_blocks=8,
        dec_blocks=[4, 2, 2]
    )
    model.eval()
    print("[OK] AIRNet model successfully instantiated.")
    
    # 2. Generate Random Dummy Tensor (1, 1, 128, 128)
    dummy_input = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    print(f"[OK] Created dummy input tensor: {tuple(dummy_input.shape)}")

    # 3. Execute Full Forward Pass
    with torch.no_grad():
        out_dict = model(dummy_input)

    restored = out_dict["restored"]
    edge = out_dict["edge"]
    noise_score = out_dict["noise"]
    blur_score = out_dict["blur"]
    texture_score = out_dict["texture"]

    print("\n--- OUTPUT FORWARD PASS RESULTS ---")
    print(f"Restored Output Shape: {tuple(restored.shape)}")
    print(f"Edge Output Shape:     {tuple(edge.shape)}")
    print(f"Predicted Noise Score: {noise_score.item():.4f} ({noise_score.item() * 100:.1f}%)")
    print(f"Predicted Blur Score:  {blur_score.item():.4f} ({blur_score.item() * 100:.1f}%)")
    print(f"Predicted Texture Score: {texture_score.item():.4f} ({texture_score.item() * 100:.1f}%)")

    # 4. Strict Assertions
    assert restored.shape == (1, 1, 256, 256), f"Expected Restored shape (1, 1, 256, 256), got {restored.shape}"
    assert edge.shape == (1, 1, 256, 256), f"Expected Edge shape (1, 1, 256, 256), got {edge.shape}"
    assert 0.0 <= noise_score.item() <= 1.0, f"Noise score out of bounds: {noise_score.item()}"
    assert 0.0 <= blur_score.item() <= 1.0, f"Blur score out of bounds: {blur_score.item()}"
    assert 0.0 <= texture_score.item() <= 1.0, f"Texture score out of bounds: {texture_score.item()}"
    print("[OK] All shape and scalar range assertions PASSED successfully!")

    # 5. Test Edge Target Generator & AIRNetHybridLoss
    dummy_gt = torch.rand(1, 1, 256, 256, dtype=torch.float32)
    gt_edges = compute_sobel_edges(dummy_gt)
    assert gt_edges.shape == (1, 1, 256, 256), f"Expected GT edges shape (1, 1, 256, 256), got {gt_edges.shape}"
    print(f"[OK] Sobel Edge Target Generator PASSED. GT Edge map shape: {tuple(gt_edges.shape)}")

    criterion = AIRNetHybridLoss(l1_weight=0.60, ssim_weight=0.25, edge_weight=0.15, data_range=1.0)
    loss_val = criterion(out_dict, dummy_gt)
    print(f"[OK] AIRNetHybridLoss Forward Pass PASSED. Calculated Loss: {loss_val.item():.6f}")

    print("====================================================")
    print("AIR-NET V1 VERIFICATION COMPLETE: ALL CHECKS PASSED!")
    print("====================================================")

if __name__ == "__main__":
    main()
