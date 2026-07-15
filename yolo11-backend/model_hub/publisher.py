"""Publica un modelo al Model Hub: cifra, crea un Release, sube los assets
cifrados y actualiza catalog.json — todo vía la API REST de GitHub, sin
necesitar git ni un working tree dentro del contenedor que publica.
"""
import base64
import json
import re
import tempfile
from pathlib import Path
from typing import Optional

import requests

from . import catalog as cat
from . import crypto
from .config import ModelHubConfig

API = "https://api.github.com"


def _slugify(name: str) -> str:
    """Nombre elegido por el usuario -> nombre de archivo seguro. Se le quita
    cualquier extensión que ya traiga (el nombre es un label libre, no dicta
    el formato real del artifact)."""
    base = Path(name).stem or name
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", base).strip("-")
    return slug or "model"


class PublishError(Exception):
    pass


class ModelHubPublisher:
    def __init__(self, config: Optional[ModelHubConfig] = None):
        self.cfg = config or ModelHubConfig.from_env()

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        token = self.cfg.require_gh_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _repo_url(self, path: str) -> str:
        return f"{API}/repos/{self.cfg.repo}{path}"

    # ── Catálogo ──────────────────────────────────────────────────────────────

    def _get_catalog(self) -> tuple[dict, Optional[str]]:
        """Devuelve (catalog_dict, sha_del_archivo_en_github) para poder
        actualizarlo después sin pisar una escritura concurrente.

        La API de Contents de GitHub omite el campo 'content' (encoding
        pasa a "none") para archivos mayores a 1MB — devolvía
        JSONDecodeError('Expecting value: line 1 column 1') porque
        base64.b64decode("") -> b"" -> json.loads("") revienta. Con las
        series de métricas ya no viven inline en catalog.json (ver
        publish(): rows_url) esto no debería volver a pasar en la práctica,
        pero se deja el fallback al Git Data API (blobs, soporta hasta
        100MB) por las dudas — es la forma correcta de leer un archivo
        grande de todos modos."""
        resp = requests.get(self._repo_url("/contents/catalog.json"), headers=self._headers(), timeout=15)
        if resp.status_code == 404:
            return cat.empty_catalog(), None
        resp.raise_for_status()
        body = resp.json()
        sha = body["sha"]
        if body.get("content"):
            content = base64.b64decode(body["content"]).decode("utf-8")
        else:
            blob_resp = requests.get(self._repo_url(f"/git/blobs/{sha}"), headers=self._headers(), timeout=30)
            blob_resp.raise_for_status()
            content = base64.b64decode(blob_resp.json()["content"]).decode("utf-8")
        return json.loads(content), sha

    def _put_repo_file(self, path: str, content: bytes, message: str) -> str:
        """Commitea un archivo chico directo al repo (Contents API) y devuelve
        su URL raw.githubusercontent.com. A diferencia de los assets de
        Release (usados para .pt/.enc/.onnx.enc, que sí pueden pesar cientos
        de MB), estos archivos son livianos (rows.json, KBs) y se benefician
        de vivir en el repo: raw.githubusercontent.com tiene CORS abierto
        (access-control-allow-origin: *) — los assets de Release NO, así que
        un fetch() desde el sitio del hub los bloquea silenciosamente (sin
        error visible, el gráfico simplemente no aparece)."""
        payload = {
            "message": message,
            "content": base64.b64encode(content).decode("ascii"),
        }
        resp = requests.put(self._repo_url(f"/contents/{path}"), headers=self._headers(), json=payload, timeout=30)
        if resp.status_code >= 300:
            raise PublishError(f"No se pudo commitear {path}: {resp.status_code} {resp.text[:300]}")
        return f"https://raw.githubusercontent.com/{self.cfg.repo}/main/{path}"

    def _put_catalog(self, catalog: dict, sha: Optional[str]) -> None:
        cat.touch(catalog)
        payload = {
            "message": f"chore(catalog): actualizar catalog.json ({cat.new_version_id()})",
            "content": base64.b64encode(json.dumps(catalog, indent=2, ensure_ascii=False).encode("utf-8")).decode("ascii"),
        }
        if sha:
            payload["sha"] = sha
        resp = requests.put(self._repo_url("/contents/catalog.json"), headers=self._headers(), json=payload, timeout=15)
        if resp.status_code >= 300:
            raise PublishError(f"No se pudo actualizar catalog.json: {resp.status_code} {resp.text[:300]}")

    # ── Release ───────────────────────────────────────────────────────────────

    def _create_release(self, tag: str, name: str, body: str) -> dict:
        resp = requests.post(
            self._repo_url("/releases"),
            headers=self._headers(),
            json={"tag_name": tag, "name": name, "body": body, "draft": False, "prerelease": False},
            timeout=20,
        )
        if resp.status_code >= 300:
            raise PublishError(f"No se pudo crear el release {tag}: {resp.status_code} {resp.text[:300]}")
        return resp.json()

    def _upload_asset(self, upload_url_template: str, filename: str, data: bytes) -> dict:
        upload_url = upload_url_template.split("{")[0] + f"?name={filename}"
        headers = self._headers()
        headers["Content-Type"] = "application/octet-stream"
        resp = requests.post(upload_url, headers=headers, data=data, timeout=120)
        if resp.status_code >= 300:
            raise PublishError(f"No se pudo subir el asset {filename}: {resp.status_code} {resp.text[:300]}")
        return resp.json()

    # ── API pública ───────────────────────────────────────────────────────────

    def publish(
        self,
        family: str,
        size: str,
        display_name: str,
        description: str,
        files: dict,
        headline_metric: dict,
        chart_series: list,
        rows: list,
        extra_metrics: Optional[list] = None,
        tags: Optional[list] = None,
        version_scheme: Optional[str] = None,
        plain_files: Optional[dict] = None,
    ) -> dict:
        """
        files: {"pt": Path(...), "onnx": Path(...)} — rutas a los archivos
               en PLANO (sin cifrar); esta función los cifra a temporales.
        plain_files: {"meta": Path(...)} — archivos que se suben SIN cifrar
               (ej: JSON de metadata compatible con HIS — no es sensible, y
               HIS/otros consumidores necesitan poder leerlo directo).
        headline_metric: {"label": "plate_acc", "value": 0.9962, "format": "percent"}
        chart_series: [{"key": "plate_acc", "label": "plate_acc"}, ...]
        rows: filas de métricas por época (lo que se grafica) — se suben como
              asset propio (rows_url), NO inline en catalog.json: con cientos/
              miles de épocas el catálogo compartido por TODOS los modelos
              podía superar 1MB y romper _get_catalog() para cualquier
              publicación futura (ver docstring de _get_catalog).
        tags: lista de strings libres (ej: ["precintos", "tapas de válvula"]
              para YOLO) — puramente informativo/filtrable, no participa de
              la identidad del modelo.
        display_name determina la identidad (model_id = family + slug(display_name)):
              publicar dos veces con el mismo nombre agrega una versión nueva
              a la MISMA entrada del catálogo en vez de crear una duplicada
              (ver catalog.make_model_id) — versions[0] siempre es la última.
        Retorna {"model_id", "version", "hub_url"}.
        """
        key = self.cfg.require_key()
        model_id = cat.make_model_id(family, display_name)
        version_id = cat.new_version_id()
        tag = f"{model_id}-{version_id}"

        release = self._create_release(tag, f"{display_name} — {version_id}", description)
        upload_url_template = release["upload_url"]

        name_slug = _slugify(display_name)
        artifacts = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt, path in (plain_files or {}).items():
                path = Path(path)
                plain = path.read_bytes()
                # "_meta.json" (no ".meta.json"): así HIS lo encuentra tanto
                # para el .pt como el .onnx (les quita la extensión antes de
                # buscar el sidecar) — un único archivo linkea ambos formatos
                # por compartir el mismo nombre base.
                filename = f"{name_slug}_meta.json" if fmt == "meta" else f"{name_slug}.{fmt}.json"
                asset = self._upload_asset(upload_url_template, filename, plain)
                artifacts.append({
                    "format": fmt,
                    "filename": filename,
                    "size_bytes": len(plain),
                    "encrypted": False,
                    "sha256_plain": crypto.sha256_bytes(plain),
                    "sha256_enc": None,
                    "download_url": asset["browser_download_url"],
                })
            for fmt, path in files.items():
                path = Path(path)
                plain = path.read_bytes()
                enc_bytes = crypto.encrypt_bytes(plain, key)
                filename = f"{name_slug}.{fmt}.enc"
                asset = self._upload_asset(upload_url_template, filename, enc_bytes)
                artifacts.append({
                    "format": fmt,
                    "filename": filename,
                    "size_bytes": len(enc_bytes),
                    "encrypted": True,
                    "sha256_plain": crypto.sha256_bytes(plain),
                    "sha256_enc": crypto.sha256_bytes(enc_bytes),
                    "download_url": asset["browser_download_url"],
                })

        # rows commiteadas al repo (no a un asset de Release — esos no tienen
        # CORS, el fetch() del sitio del hub los bloquea sin error visible;
        # tampoco inline en catalog.json, ver docstring de _get_catalog).
        rows_bytes = json.dumps(rows, ensure_ascii=False).encode("utf-8")
        rows_path = f"metrics/{model_id}/{version_id}.json"
        rows_url = self._put_repo_file(rows_path, rows_bytes, f"chore(metrics): {model_id} {version_id}")

        catalog, sha = self._get_catalog()
        model = cat.upsert_model(
            catalog, model_id, family, size, display_name,
            tags=tags, version_scheme=version_scheme,
        )
        version_entry = {
            "version": version_id,
            "description": description,
            "released_at": release["created_at"],
            "size": size,
            "tags": tags or [],
            "headline_metric": headline_metric,
            "extra_metrics": extra_metrics or [],
            "chart_series": chart_series,
            "rows_url": rows_url,
            "rows_count": len(rows),
            "artifacts": artifacts,
        }
        cat.add_version(model, version_entry)
        self._put_catalog(catalog, sha)

        hub_url = f"https://oneclickeb.github.io/models-hub/#model={model_id}"
        return {"model_id": model_id, "version": version_id, "hub_url": hub_url}

    def add_artifact(
        self, model_id: str, version: str, fmt: str, file_path, plain: bool = False
    ) -> dict:
        """Agrega un artifact NUEVO a una versión YA publicada (ej: subiste
        `precintos-s.pt` y más tarde generaste el `.onnx` o el `.dlc` que en
        su momento no tenías). Sube el archivo al mismo Release ya existente
        (mismo tag `model_id-version`) y actualiza `catalog.json` in-place —
        no crea una versión nueva ni toca los artifacts que ya estaban.

        `plain=True` para archivos que no deben cifrarse (igual que
        `plain_files` en `publish()`, ej. un metadata JSON).

        Si ya existe un artifact con el mismo `fmt` en esa versión, se
        REEMPLAZA (mismo criterio que "el catálogo es mutable en formatos,
        inmutable en identidad/versión").
        """
        tag = f"{model_id}-{version}"
        rel_resp = requests.get(self._repo_url(f"/releases/tags/{tag}"), headers=self._headers(), timeout=15)
        if rel_resp.status_code != 200:
            raise PublishError(f"No se encontró el release {tag} — ¿model_id/version correctos?")
        release = rel_resp.json()

        catalog, sha = self._get_catalog()
        model = cat.find_model(catalog, model_id)
        if model is None:
            raise PublishError(f"Modelo {model_id} no encontrado en el catálogo")
        version_entry = next((v for v in model["versions"] if v["version"] == version), None)
        if version_entry is None:
            raise PublishError(f"Versión {version} no encontrada para {model_id}")

        path = Path(file_path)
        plain_bytes = path.read_bytes()
        name_slug = _slugify(model["display_name"])

        if plain:
            filename = f"{name_slug}_meta.json" if fmt == "meta" else f"{name_slug}.{fmt}.json"
            data, encrypted, sha_enc = plain_bytes, False, None
        else:
            key = self.cfg.require_key()
            filename = f"{name_slug}.{fmt}.enc"
            data = crypto.encrypt_bytes(plain_bytes, key)
            encrypted, sha_enc = True, crypto.sha256_bytes(data)

        asset = self._upload_asset(release["upload_url"], filename, data)
        artifact = {
            "format": fmt,
            "filename": filename,
            "size_bytes": len(data),
            "encrypted": encrypted,
            "sha256_plain": crypto.sha256_bytes(plain_bytes),
            "sha256_enc": sha_enc,
            "download_url": asset["browser_download_url"],
        }
        version_entry["artifacts"] = [
            a for a in version_entry.get("artifacts", []) if a.get("format") != fmt
        ] + [artifact]
        self._put_catalog(catalog, sha)
        return artifact
