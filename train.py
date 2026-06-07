import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.dual_branch import DualBranchModel
from src.preprocessing.dataset_builder import create_dataloaders
from src.preprocessing.srm_filters import SRMFilterLayer
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def build_model(config: dict) -> DualBranchModel:
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    return DualBranchModel(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        pretrained_rgb=model_cfg.get("pretrained_rgb", True),
        pretrained_noise=model_cfg.get("pretrained_noise", False),
        feature_dim=model_cfg.get("feature_dim", 1280),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        dropout=model_cfg.get("dropout", 0.3),
        freeze_bn=train_cfg.get("freeze_bn", True),
    )



def main() -> None:
    # ── Audit dataset for missing/corrupt files before training ──────────────
    import subprocess, sys
    audit_result = subprocess.run(
        [sys.executable, "scripts/audit_dataset.py", "--fix"],
        capture_output=False,
    )
    if audit_result.returncode != 0:
        # Exit code 1 means bad rows were found and removed.
        # On Colab this is expected: the uploaded CSV has local/Windows paths
        # that don't exist on Colab. They are removed here; create_dataloaders()
        # will rebuild the CSV from the raw data directory.
        print("\n[train.py] Dataset audit removed stale/bad rows from split_metadata.csv.")
        print("[train.py] Continuing — create_dataloaders() will rebuild from raw data.\n")

    parser = argparse.ArgumentParser(description="Train the DeepSight model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip phase-1 (noise branch warm-up)",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run only phase-1 then exit",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    paths_cfg = config.get("paths", {})

    seed = data_cfg.get("seed") or train_cfg.get("seed")
    if seed is not None:
        seed_everything(int(seed))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(config)
    srm_layer = SRMFilterLayer()

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    # SRM layer stays on CPU when num_workers > 0 (workers can't share GPU tensors)
    if num_workers == 0 and device.type == "cuda":
        srm_layer = srm_layer.to(device)

    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=batch_size,
        num_workers=num_workers,
        srm_layer=srm_layer,
    )

    logger.info(
        "Dataset sizes: train=%d val=%d test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Training or validation split is empty. Check raw data paths.")

    trainer = Trainer(
        model,
        config,
        device,
        checkpoint_dir=paths_cfg.get("checkpoint_dir", "models/checkpoints"),
    )

    if args.resume is not None:
        trainer.load_checkpoint(str(args.resume))

    run_phase1 = train_cfg.get("phase1_epochs", 0) > 0 and not args.skip_phase1
    if run_phase1:
        logger.info("Phase 1: training noise branch + heads")
        set_trainable(model.rgb_branch, False)
        set_trainable(model.noise_branch, True)
        set_trainable(model.fusion, True)
        set_trainable(model.classification_head, True)
        trainer.fit(train_loader, val_loader, phase1=True)
        if args.phase1_only:
            return

    logger.info("Phase 2: training full model")
    set_trainable(model.rgb_branch, True)
    set_trainable(model.noise_branch, True)
    set_trainable(model.fusion, True)
    set_trainable(model.classification_head, True)

    # Reset optimiser for Phase 2 only when transitioning from Phase 1
    # (not when resuming from a Phase 2 checkpoint)
    if trainer.current_epoch <= train_cfg.get("phase1_epochs", 20):
        trainer.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]) * 0.5,
            weight_decay=float(train_cfg["weight_decay"]),
        )
        from src.training.scheduler import build_scheduler
        trainer.scheduler = build_scheduler(trainer.optimizer, trainer.cfg)

    trainer.fit(train_loader, val_loader, phase1=False)


if __name__ == "__main__":
    main()
