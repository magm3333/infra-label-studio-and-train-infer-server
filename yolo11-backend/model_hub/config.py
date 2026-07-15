"""Configuración del driver, leída de variables de entorno.

Ningún valor tiene un default hardcodeado para la clave o el token — si
faltan, las operaciones que los necesiten fallan explícitamente en vez de
usar un placeholder inseguro.

La clave (`MODEL_HUB_KEY`) admite dos formas:
- `MODEL_HUB_KEY_FILE=/ruta/al/archivo` (preferida): evita pasar passphrases
  con caracteres especiales (`$`, `#`, etc.) por interpolación de variables
  de docker-compose, que puede corromperlas silenciosamente.
- `MODEL_HUB_KEY=valor` directo: solo para passphrases sin caracteres que
  choquen con la sintaxis `${...}` de compose/shell.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelHubConfig:
    repo: str
    key: Optional[str]
    gh_token: Optional[str]
    catalog_url: str

    @classmethod
    def from_env(cls) -> "ModelHubConfig":
        repo = os.environ.get("MODEL_HUB_REPO", "OneclickEB/models-hub")
        key_file = os.environ.get("MODEL_HUB_KEY_FILE")
        if key_file and Path(key_file).exists():
            key = Path(key_file).read_text().strip() or None
        else:
            key = os.environ.get("MODEL_HUB_KEY") or None
        return cls(
            repo=repo,
            key=key,
            gh_token=os.environ.get("MODEL_HUB_GH_TOKEN") or None,
            catalog_url=os.environ.get(
                "MODEL_HUB_CATALOG_URL",
                f"https://raw.githubusercontent.com/{repo}/main/catalog.json",
            ),
        )

    def require_key(self) -> str:
        if not self.key:
            raise RuntimeError(
                "MODEL_HUB_KEY no está configurada — no se puede cifrar/descifrar"
            )
        return self.key

    def require_gh_token(self) -> str:
        if not self.gh_token:
            raise RuntimeError(
                "MODEL_HUB_GH_TOKEN no está configurada — no se puede publicar"
            )
        return self.gh_token
