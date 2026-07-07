from .dataloader import (
    MorphologyRecord,
    PulsePPGMorphologyDataset,
    build_morphology_records,
    load_morphology_index,
    write_morphology_index,
)


class _TorchRequired:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise ImportError(f"{self.name} requires torch. Install torch to train models.")

    def __getattr__(self, name: str):
        raise ImportError(f"{self.name} requires torch. Install torch to train models.")


try:
    from .encoder import PulsePPGResNet1D, build_pulseppg_encoder
    from .heads import MorphologyAwarePulsePPG, morphology_ssl_loss, supervised_nt_xent_loss
    from .trainer import MorphologyTrainConfig, fit_morphology_model
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    PulsePPGResNet1D = _TorchRequired("PulsePPGResNet1D")
    build_pulseppg_encoder = _TorchRequired("build_pulseppg_encoder")
    MorphologyAwarePulsePPG = _TorchRequired("MorphologyAwarePulsePPG")
    morphology_ssl_loss = _TorchRequired("morphology_ssl_loss")
    supervised_nt_xent_loss = _TorchRequired("supervised_nt_xent_loss")
    MorphologyTrainConfig = _TorchRequired("MorphologyTrainConfig")
    fit_morphology_model = _TorchRequired("fit_morphology_model")

__all__ = [
    "MorphologyAwarePulsePPG",
    "MorphologyRecord",
    "MorphologyTrainConfig",
    "PulsePPGMorphologyDataset",
    "PulsePPGResNet1D",
    "build_morphology_records",
    "build_pulseppg_encoder",
    "fit_morphology_model",
    "load_morphology_index",
    "morphology_ssl_loss",
    "supervised_nt_xent_loss",
    "write_morphology_index",
]
