import pytest

from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer


class TestBuildScheduler:
    def test_build_returns_scheduler(self):
        import torch
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        cfg = {"training": {"scheduler": "plateau", "phase2_epochs": 30, "warmup_epochs": 5}}
        scheduler = build_scheduler(optimizer, cfg)
        assert scheduler is not None
        assert hasattr(scheduler, "step")


class TestTrainerInit:
    def test_init_requires_model_and_config(self):
        import torch
        model = torch.nn.Linear(10, 1)
        config = {"training": {"learning_rate": 0.001, "phase1_epochs": 5, "phase2_epochs": 20, "scheduler": "plateau", "warmup_epochs": 3}}
        trainer = Trainer(model, config, torch.device("cpu"))
        assert trainer.model is not None
        assert trainer.device == torch.device("cpu")

    def test_optimizer_created(self):
        import torch
        model = torch.nn.Linear(10, 1)
        config = {"training": {"learning_rate": 0.001, "phase1_epochs": 5, "phase2_epochs": 20, "scheduler": "plateau", "warmup_epochs": 3}}
        trainer = Trainer(model, config, torch.device("cpu"))
        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_scheduler_created(self):
        import torch
        model = torch.nn.Linear(10, 1)
        config = {"training": {"learning_rate": 0.001, "phase1_epochs": 5, "phase2_epochs": 20, "scheduler": "plateau", "warmup_epochs": 3}}
        trainer = Trainer(model, config, torch.device("cpu"))
        assert trainer.scheduler is not None

    def test_checkpoint_dir_created(self):
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(10, 1)
            config = {"training": {"phase1_epochs": 5, "phase2_epochs": 20, "scheduler": "plateau", "warmup_epochs": 3}}
            trainer = Trainer(model, config, torch.device("cpu"), checkpoint_dir=tmp)
            import os
            assert os.path.isdir(tmp)
