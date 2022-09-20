import shutil
from pathlib import Path
from subprocess import check_call

# Install dependencies
print("Installing dependencies...")
scripts_dir = Path("code")
assert scripts_dir.exists(), f"Path {scripts_dir.absolute()} does not exist"

if not (scripts_dir / "nnunet").exists():
    check_call(["git", "clone", "https://github.com/DIAGNijmegen/nnUNet", (scripts_dir / "nnunet").as_posix()])
if not ((scripts_dir / "picai_baseline")).exists():
    check_call(["git", "clone", "https://github.com/DIAGNijmegen/picai_baseline", (scripts_dir / "picai_baseline").as_posix()])

# Move custom files to the nnU-Net installation
shutil.copy(scripts_dir / "picai_baseline/src/picai_baseline/nnunet/training_docker/io.py",
            scripts_dir / "nnunet/nnunet/utilities/io.py")
shutil.copy(scripts_dir / "picai_baseline/src/picai_baseline/nnunet/training_docker/nnUNetTrainerV2_focalLoss.py",
            scripts_dir / "nnunet/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_focalLoss.py")
shutil.copy(scripts_dir / "picai_baseline/src/picai_baseline/nnunet/training_docker/nnUNetTrainerV2_Loss_CE_checkpoints.py",
            scripts_dir / "nnunet/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_Loss_CE_checkpoints.py")
shutil.copy(scripts_dir / "picai_baseline/src/picai_baseline/nnunet/training_docker/nnUNetTrainerV2_Loss_FL_and_CE.py",
            scripts_dir / "nnunet/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_Loss_FL_and_CE.py")
shutil.copy(scripts_dir / "nnunet_custom_setup.py",
            scripts_dir / "nnunet/setup.py")
