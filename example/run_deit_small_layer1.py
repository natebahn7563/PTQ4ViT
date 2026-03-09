"""
PTQ4ViT - DeiT-Small W8A8 Quantization (테스트: 첫 번째 블록만)
실행 위치: PTQ4ViT/ 루트에서
  python example/run_deit_small.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import timm
import torchvision
import torchvision.transforms as transforms

import configs.PTQ4ViT as ptq_cfg
from utils.net_wrap import wrap_certain_modules_in_net
from utils.quant_calib import HessianQuantCalibrator

# ── 설정 ──────────────────────────────────────────────
MODEL_PT  = "/scratch/x3433a06/weights/deit_small.pt"
SAVE_PATH = "/scratch/x3433a06/quantized_weights/deit_ptq4vit_w8a8.pt"
CIFAR_DIR = "/scratch/x3433a06/cifar10"
N_CALIB   = 32
DEVICE    = "cuda"
# ─────────────────────────────────────────────────────


def get_cifar_calib_loader():
    """CIFAR-10을 224x224로 resize해서 calibration 데이터로 사용 (자동 다운로드)"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=CIFAR_DIR, train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=N_CALIB, shuffle=True, num_workers=4
    )
    return loader


def analyze_4msb_all(quantized_weights):
    total_0000 = total_1111 = total_all = 0

    for name, w in quantized_weights.items():
        w_int = w.to(torch.int8)
        msb4  = (w_int.view(torch.uint8) >> 4) & 0xF
        n     = w_int.numel()
        c0    = (msb4 == 0b0000).sum().item()
        c1    = (msb4 == 0b1111).sum().item()

        total_0000 += c0
        total_1111 += c1
        total_all  += n

        print(f"  {name:60s}| 0000={c0/n:.3f}  1111={c1/n:.3f}  combined={(c0+c1)/n:.3f}")

    print(f"\n  [TOTAL] 0000={total_0000/total_all:.4f}  "
          f"1111={total_1111/total_all:.4f}  "
          f"combined={(total_0000+total_1111)/total_all:.4f}")


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # 1. FP32 모델 로드
    print("[1/4] Loading FP32 model...")
    model = timm.create_model("deit_small_patch16_224", pretrained=False)
    state_dict = torch.load(MODEL_PT, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    print("      Done.")

    # 2. 첫 번째 블록(block.0)만 wrapping
    print("[2/4] Wrapping quantization layers (block 0 only)...")
    layers = [0]  # 첫 번째 블록만
    modules_to_wrap = ["qkv", "proj", "fc1", "fc2"]  # Conv2d 제외, Linear만
    wrapped_modules = wrap_certain_modules_in_net(model, ptq_cfg, layers, modules_to_wrap)
    print(f"      Wrapped {len(wrapped_modules)} modules.")

    # 3. Calibration
    print(f"[3/4] Calibrating with {N_CALIB} CIFAR-10 images (resized 224x224)...")
    calib_loader = get_cifar_calib_loader()
    calibrator = HessianQuantCalibrator(
        net=model,
        wrapped_modules=wrapped_modules,
        calib_loader=calib_loader,
        sequential=False,
        batch_size=1,
    )
    calibrator.batching_quant_calib()
    print("      Done.")

    # 4. Quantized weights 저장
    print("[4/4] Saving quantized weights...")
    target_names = ["qkv", "proj", "fc1", "fc2"]
    quantized_weights = {}
    for name, module in model.named_modules():
        if any(t in name for t in target_names):
            if hasattr(module, "weight") and module.weight is not None:
                quantized_weights[name] = module.weight.data.cpu()

    torch.save(quantized_weights, SAVE_PATH)
    print(f"      Saved {len(quantized_weights)} layers → {SAVE_PATH}")

    # 4MSB 분석
    print("\n[4MSB Analysis]")
    analyze_4msb_all(quantized_weights)


if __name__ == "__main__":
    main()