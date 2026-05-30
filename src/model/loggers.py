from __future__ import annotations

from typing import Any, Protocol


class Logger(Protocol):
    def log_scalars(self, metrics: dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


class TensorBoardLogger:
    def __init__(self, log_dir: str, run_name: str = "") -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.run_name = run_name
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class WandbLogger:
    def __init__(
        self,
        project: str,
        run_name: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config or {})

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self.run.finish()


class StdoutLogger:
    """No-op-ish logger that prints to stdout. Useful for smoke tests and CI."""

    def __init__(self, run_name: str = "") -> None:
        self.run_name = run_name

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        items = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[{self.run_name} step={step}] {items}")

    def close(self) -> None:
        pass
