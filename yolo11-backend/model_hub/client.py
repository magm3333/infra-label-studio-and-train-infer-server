"""Cliente de solo lectura del Model Hub: listar, seleccionar y descargar
(desencriptando de forma transparente) modelos publicados.

No requiere token de GitHub — el catálogo y los assets de Release son
públicos. Solo requiere MODEL_HUB_KEY para poder desencriptar.
"""
from pathlib import Path
from typing import Optional

import requests

from . import crypto
from .config import ModelHubConfig


class ModelNotFoundError(Exception):
    pass


class ModelHubClient:
    def __init__(self, config: Optional[ModelHubConfig] = None):
        self.cfg = config or ModelHubConfig.from_env()
        self._catalog_cache: Optional[dict] = None

    def _catalog(self, refresh: bool = False) -> dict:
        if self._catalog_cache is None or refresh:
            resp = requests.get(self.cfg.catalog_url, timeout=15, headers={"Cache-Control": "no-cache"})
            resp.raise_for_status()
            self._catalog_cache = resp.json()
        return self._catalog_cache

    def list_models(self, family: Optional[str] = None) -> list:
        models = self._catalog(refresh=True).get("models", [])
        if family:
            models = [m for m in models if m.get("family") == family]
        return models

    def get_model(self, model_id: str) -> dict:
        for m in self._catalog().get("models", []):
            if m.get("model_id") == model_id:
                return m
        raise ModelNotFoundError(f"Modelo no encontrado en el catálogo: {model_id}")

    def get_version(self, model_id: str, version: str = "latest") -> dict:
        model = self.get_model(model_id)
        versions = model.get("versions", [])
        if not versions:
            raise ModelNotFoundError(f"{model_id} no tiene versiones publicadas")
        if version == "latest":
            return versions[0]  # el publisher inserta al principio (más reciente primero)
        for v in versions:
            if v.get("version") == version:
                return v
        raise ModelNotFoundError(f"Versión {version} no encontrada para {model_id}")

    def get_rows(self, version: dict) -> list:
        """Filas de métricas por época de una versión (para graficar). Se
        suben como asset propio (rows_url) en vez de vivir inline en
        catalog.json — ver publisher.py. Compatible con entradas viejas que
        todavía tengan 'rows' inline (previas a este cambio)."""
        if "rows" in version:
            return version["rows"]
        rows_url = version.get("rows_url")
        if not rows_url:
            return []
        resp = requests.get(rows_url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def download(
        self,
        model_id: str,
        version: str = "latest",
        format: str = "pt",
        dest_dir: str = ".",
    ) -> Path:
        """Descarga, verifica integridad y desencripta (si corresponde).
        Devuelve la ruta del archivo listo para usar.

        Artifacts con `encrypted: false` (ej. el JSON de metadata estilo HIS)
        no requieren MODEL_HUB_KEY — se descargan y verifican tal cual."""
        v = self.get_version(model_id, version)
        artifact = next((a for a in v.get("artifacts", []) if a.get("format") == format), None)
        if artifact is None:
            available = [a.get("format") for a in v.get("artifacts", [])]
            raise ModelNotFoundError(f"Formato '{format}' no disponible para {model_id} (hay: {available})")

        resp = requests.get(artifact["download_url"], timeout=120)
        resp.raise_for_status()
        raw = resp.content
        is_encrypted = artifact.get("encrypted", True)

        if is_encrypted:
            if crypto.sha256_bytes(raw) != artifact.get("sha256_enc"):
                raise crypto.DecryptionError(
                    "sha256 del archivo descargado no coincide — descarga corrupta"
                )
            plain = crypto.decrypt_bytes(raw, self.cfg.require_key())
        else:
            plain = raw

        if crypto.sha256_bytes(plain) != artifact.get("sha256_plain"):
            raise crypto.DecryptionError(
                "sha256 no coincide — clave incorrecta o archivo corrupto"
            )

        dest_dir_path = Path(dest_dir)
        dest_dir_path.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir_path / f"{model_id}-{v['version']}.{format}"
        dest_path.write_bytes(plain)
        return dest_path
