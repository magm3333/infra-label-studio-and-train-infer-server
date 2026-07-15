"""Estructura y manejo de catalog.json.

Se trabaja con dicts planos (no dataclasses) para no pelear con la
serialización JSON — el esquema está documentado acá y en
plan-model-hub.md. Ver también models-hub/README.md.
"""
import datetime as _dt
import re as _re
import unicodedata as _ud
from typing import Optional

CATALOG_VERSION = "1"


def _slug_part(text: str) -> str:
    """Normaliza un componente de model_id (usado también como git tag del
    Release en publisher.py, que no acepta espacios/acentos/símbolos —
    ver 'tag_name is not a valid tag' si se le pasa texto libre crudo, ej.
    un target como 'precintos y tapas de válvula' escrito por el usuario)."""
    text = _ud.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = _re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "x"


def empty_catalog() -> dict:
    return {
        "catalog_version": CATALOG_VERSION,
        "updated_at": _now_iso(),
        "models": [],
    }


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_version_id() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def make_model_id(family: str, display_name: str) -> str:
    """Identidad estable de una entrada del catálogo, derivada del NOMBRE que
    elige el usuario al publicar (no de size/target/task) — así "mismo nombre"
    es literalmente "mismo model_id": una segunda publicación con el mismo
    nombre agrega una versión nueva a la MISMA entrada (más reciente primero,
    ver add_version) en vez de crear una entrada duplicada. Antes se derivaba
    de (family, size, target-o-task) — un target de texto libre re-tipeado
    ligeramente distinto entre subidas (ej. 'precintos' vs 'precintos y tapas
    de válvula') producía silenciosamente identidades distintas.

    Ej: family='lprocr', display_name='nv-lpr-n-v2.pt' -> 'lprocr-nv-lpr-n-v2-pt'
        family='yolo', display_name='precintos-s.pt' -> 'yolo-precintos-s-pt'
    """
    return "-".join([_slug_part(family), _slug_part(display_name)])


def find_model(catalog: dict, model_id: str) -> Optional[dict]:
    for m in catalog.get("models", []):
        if m.get("model_id") == model_id:
            return m
    return None


def upsert_model(
    catalog: dict,
    model_id: str,
    family: str,
    size: str,
    display_name: str,
    tags: Optional[list] = None,
    version_scheme: Optional[str] = None,
) -> dict:
    """Devuelve la entrada del modelo, creándola si no existe. Si ya existe
    (mismo model_id -> mismo nombre), se refrescan size/tags/version_scheme
    con los de esta publicación (la más reciente manda para estos campos de
    metadata; el historial de versiones en sí es aditivo, ver add_version)."""
    model = find_model(catalog, model_id)
    if model is None:
        model = {
            "model_id": model_id,
            "family": family,
            "size": size,
            "tags": tags or [],
            "version_scheme": version_scheme,
            "display_name": display_name,
            "versions": [],
        }
        catalog.setdefault("models", []).append(model)
    else:
        model["size"] = size
        model["tags"] = tags or []
        model["version_scheme"] = version_scheme
        model["display_name"] = display_name
    return model


def add_version(model: dict, version_entry: dict) -> None:
    """Agrega una versión al principio (más reciente primero)."""
    model.setdefault("versions", []).insert(0, version_entry)


def touch(catalog: dict) -> None:
    catalog["updated_at"] = _now_iso()
