import re
from importlib import metadata

import torch


def _normalize_version(version):
    return version.split("+", 1)[0]


def _get_installed_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _get_required_torch_version_for_torchvision():
    try:
        requirements = metadata.distribution("torchvision").requires or []
    except metadata.PackageNotFoundError:
        return None

    for requirement in requirements:
        if not requirement.lower().startswith("torch "):
            continue

        match = re.search(r"==\s*([0-9]+(?:\.[0-9]+){1,2}(?:\+[A-Za-z0-9_.-]+)?)", requirement)
        if match:
            return match.group(1)

    return None


def check_torchvision_compatibility():
    torch_version = _get_installed_version("torch")
    torchvision_version = _get_installed_version("torchvision")

    if not torch_version or not torchvision_version:
        return

    required_torch_version = _get_required_torch_version_for_torchvision()
    if not required_torch_version:
        return

    if _normalize_version(torch_version) != _normalize_version(required_torch_version):
        raise RuntimeError(
            "Incompatible PyTorch install detected: "
            f"torch=={torch_version}, torchvision=={torchvision_version}. "
            f"This torchvision build expects torch=={required_torch_version}. "
            "This project does not use torchvision. Uninstall torchvision from this venv "
            "or reinstall matching torch/torchvision versions, then run the script again."
        )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
