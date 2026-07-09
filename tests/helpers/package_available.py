import platform

import pkg_resources

# In PL 2.x, xla_device module was removed
# We check for TPU availability differently now
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    _TPU_AVAILABLE = xm.xla_device_hw() == "TPU"
except ImportError:
    _TPU_AVAILABLE = False


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment."""
    try:
        return pkg_resources.require(package_name) is not None
    except pkg_resources.DistributionNotFound:
        return False


_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")

_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _package_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _package_available("fairscale")

_WANDB_AVAILABLE = _package_available("wandb")
_NEPTUNE_AVAILABLE = _package_available("neptune")
_COMET_AVAILABLE = _package_available("comet_ml")
_MLFLOW_AVAILABLE = _package_available("mlflow")