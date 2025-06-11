import os
import subprocess
import argparse
import torch

input_dir="input"
output_dir="output"

def nnunet_inference(
    input_dir,
    output_dir,
    model,
    config,
    fold,
    trainer,
    plan,
    checkpoint,
    device,
    nnunet_raw,
    nnunet_preprocessed,
    nnunet_results
):
    # Set nnUNet environment variables (important for Docker)
    
    env = os.environ.copy()
    env["input_dir"]="input"
    env["output_dir"]="output"
    env["nnUNet_raw"] = r"/app/nnUNet_data/nnUNet_raw"
    env["nnUNet_preprocessed"] = r"/app/nnUNet_data/nnUNet_preprocessed"
    env["nnUNet_results"] = r"/app/nnUNet_data/nnUNet_results"


    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", model,
        "-c", config,
        "-f", str(fold),
        "-tr", trainer,
        "-p", plan,
        "-chk", checkpoint,
        "--disable_tta",
        "-device", device
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    print("Segmentation done! Output at:", output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="nnUNetv2 Dockerized Inference")

    parser.add_argument("--input_dir",default="input", help="Input directory for images")
    parser.add_argument("--output_dir",default="output", help="Output directory for segmentations")
    parser.add_argument("--model", default="Dataset112_DentalSegmentator_v100", help="Model name")
    parser.add_argument("--config", default="3d_fullres", help="Configuration")
    parser.add_argument("--fold", default="0", help="Fold")
    parser.add_argument("--trainer", default="nnUNetTrainer", help="Trainer")
    parser.add_argument("--plan", default="nnUNetPlans", help="Plan")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or device id")
    parser.add_argument("--nnunet_raw", default="/nnUNet_data/nnUNet_raw", help="nnUNet raw data dir")
    parser.add_argument("--nnunet_preprocessed", default="/nnUNet_data/nnUNet_preprocessed", help="nnUNet preprocessed data dir")
    parser.add_argument("--nnunet_results", default="/nnUNet_data/nnUNet_results", help="nnUNet results dir")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # If device not specified, auto-detect
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    nnunet_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        config=args.config,
        fold=args.fold,
        trainer=args.trainer,
        plan=args.plan,
        checkpoint=args.checkpoint,
        device=args.device,
        nnunet_raw=args.nnunet_raw,
        nnunet_preprocessed=args.nnunet_preprocessed,
        nnunet_results=args.nnunet_results
    )
