"""model_hub — driver único de acceso al Model Hub (listar, descargar con
desencriptado transparente, publicar). Ver README.md para instrucciones de
integración."""
from .catalog import make_model_id, new_version_id
from .client import ModelHubClient, ModelNotFoundError
from .config import ModelHubConfig
from .crypto import DecryptionError
from .publisher import ModelHubPublisher, PublishError

__all__ = [
    "ModelHubClient",
    "ModelHubPublisher",
    "ModelHubConfig",
    "ModelNotFoundError",
    "PublishError",
    "DecryptionError",
    "make_model_id",
    "new_version_id",
]

__version__ = "1.0.0"
