import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import os as _os
import base64
import csv
import hashlib
import hmac
import html
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import requests
import yaml
from flask import Flask, abort, jsonify, request, send_file
from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION


app = Flask(__name__)

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

MODEL_PATH = os.getenv("MODEL_PATH", "/models/precintos-s.pt")
TRAIN_MODEL_PATH = os.getenv("TRAIN_MODEL_PATH", MODEL_PATH)
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
LABEL_STUDIO_AUTH_SCHEME = os.getenv("LABEL_STUDIO_AUTH_SCHEME", "Bearer")
LABEL_STUDIO_JWT_USER_ID = os.getenv("LABEL_STUDIO_JWT_USER_ID", "1")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "50"))
TRAIN_IMGSZ = int(os.getenv("TRAIN_IMGSZ", "640"))
TRAIN_BATCH = int(os.getenv("TRAIN_BATCH", "8"))
TRAIN_PATIENCE = int(os.getenv("TRAIN_PATIENCE", "20"))
TRAIN_WORKERS = int(os.getenv("TRAIN_WORKERS", "2"))
TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "auto")

DATA_DIR = Path("/app/data")
RUNS_DIR = DATA_DIR / "runs"
TRAIN_DIR = DATA_DIR / "training"
JOBS_PATH = DATA_DIR / "jobs.json"
LS_DATA_DIR = Path("/label-studio/data")
LS_ENV_PATH = LS_DATA_DIR / ".env"
generated_access_token = {"token": "", "exp": 0}

state = {
    "model_path": MODEL_PATH,
    "from_name": None,
    "to_name": None,
    "value": "image",
    "labels": [],
    "training": False,
    "last_train_status": None,
}
model = YOLO(state["model_path"])
lock = threading.Lock()
jobs = {}


def save_jobs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {job_id: public_job(job) for job_id, job in jobs.items()}
    JOBS_PATH.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))


def load_jobs():
    if JOBS_PATH.exists():
        try:
            data = json.loads(JOBS_PATH.read_text())
            jobs.update(data)
            changed = False
            for job in jobs.values():
                if job.get("status") in ("running", "queued"):
                    job.update({
                        "status": "interrupted",
                        "phase": "interrupted",
                        "message": "El backend se reinicio antes de que este job terminara",
                        "finished_at": job.get("finished_at") or time.time(),
                    })
                    changed = True
            if changed:
                save_jobs()
            return
        except Exception as exc:
            app.logger.warning("Could not load jobs.json: %s", exc)
    bootstrap_jobs_from_models()
    save_jobs()


def bootstrap_jobs_from_models():
    models_dir = DATA_DIR / "models"
    if not models_dir.exists():
        return
    for path in sorted(models_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime):
        metadata = read_model_metadata(path)
        if metadata:
            job_id = metadata.get("id") or f"model-{path.stem}"
            jobs[job_id] = metadata
            jobs[job_id].setdefault("id", job_id)
            jobs[job_id].setdefault("status", "completed")
            jobs[job_id].setdefault("phase", "loaded_from_saved_model")
            jobs[job_id].setdefault("trained_model", str(path))
            jobs[job_id].setdefault("created_at", path.stat().st_mtime)
            jobs[job_id].setdefault("finished_at", path.stat().st_mtime)


def load_model(path):
    global model
    with lock:
        model = YOLO(path)
        state["model_path"] = path


def parse_label_config(label_config):
    if not label_config:
        return

    import re

    rect_match = re.search(r'<RectangleLabels[^>]*name="([^"]+)"[^>]*toName="([^"]+)"', label_config)
    image_match = re.search(r'<Image[^>]*name="([^"]+)"', label_config)
    labels = re.findall(r'<Label[^>]*value="([^"]+)"', label_config)

    if rect_match:
        state["from_name"] = rect_match.group(1)
        state["to_name"] = rect_match.group(2)
    if image_match:
        state["value"] = image_match.group(1)
    if labels:
        state["labels"] = labels


def resolve_image_path(image_value):
    if not image_value:
        raise ValueError("Task has no image field")

    parsed = urlparse(image_value)
    if parsed.scheme in ("http", "https"):
        fd, path = tempfile.mkstemp(suffix=Path(parsed.path).suffix or ".jpg")
        os.close(fd)
        headers = auth_headers()
        response = requests.get(image_value, headers=headers, timeout=60)
        response.raise_for_status()
        Path(path).write_bytes(response.content)
        return path, True

    path = unquote(parsed.path or image_value)
    candidates = []
    if path.startswith("/data/"):
        relative_path = path.removeprefix("/data/")
        candidates.append(LS_DATA_DIR / relative_path)
        candidates.append(LS_DATA_DIR / "media" / relative_path)
    if path.startswith("/label-studio/data/"):
        candidates.append(Path(path))
    candidates.append(LS_DATA_DIR / path.lstrip("/"))
    candidates.append(Path(path))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate), False
    raise FileNotFoundError(f"Image not found: {image_value}; checked: {[str(candidate) for candidate in candidates]}")


def auth_headers():
    api_key = LABEL_STUDIO_API_KEY or local_label_studio_access_token()
    if not api_key:
        return {}
    scheme = LABEL_STUDIO_AUTH_SCHEME if LABEL_STUDIO_API_KEY else "Bearer"
    return {"Authorization": f"{scheme} {api_key}"}


def b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def label_studio_secret_key():
    if not LS_ENV_PATH.exists():
        return ""
    for line in LS_ENV_PATH.read_text().splitlines():
        if line.startswith("SECRET_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def local_label_studio_access_token():
    now = int(time.time())
    if generated_access_token["token"] and generated_access_token["exp"] > now + 60:
        return generated_access_token["token"]
    secret = label_studio_secret_key()
    if not secret:
        return ""
    exp = now + 3600
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "token_type": "access",
        "exp": exp,
        "iat": now,
        "jti": uuid.uuid4().hex,
        "user_id": str(LABEL_STUDIO_JWT_USER_ID),
    }
    message = ".".join([
        b64url(json.dumps(header, separators=(",", ":")).encode()),
        b64url(json.dumps(payload, separators=(",", ":")).encode()),
    ])
    signature = b64url(hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest())
    generated_access_token["token"] = f"{message}.{signature}"
    generated_access_token["exp"] = exp
    return generated_access_token["token"]


def train_device():
    return None if TRAIN_DEVICE.lower() in ("", "auto", "none") else TRAIN_DEVICE


def request_train_config(payload):
    def value(key, default, *aliases):
        header_name = f"X-Train-{key.replace('_', '-').title()}"
        header = request.headers.get(header_name)
        if header not in (None, ""):
            return header
        for candidate in (key, *aliases):
            if candidate in payload and payload[candidate] not in (None, ""):
                return payload[candidate]
        return default

    return {
        "model_path": value("model_path", TRAIN_MODEL_PATH, "model-path", "Model-Path"),
        "epochs": int(value("epochs", TRAIN_EPOCHS, "Epochs")),
        "imgsz": int(value("imgsz", TRAIN_IMGSZ, "image_size", "Image-Size", "Imgsz")),
        "batch": int(value("batch", TRAIN_BATCH, "Batch")),
        "patience": int(value("patience", TRAIN_PATIENCE, "Patience")),
        "workers": int(value("workers", TRAIN_WORKERS, "Workers")),
        "device": value("device", TRAIN_DEVICE, "Device"),
    }


def resolved_train_device(config):
    device = str(config["device"])
    return None if device.lower() in ("", "auto", "none") else device


def gpu_status():
    try:
        import torch

        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "train_device": TRAIN_DEVICE,
        }
    except Exception as exc:
        return {"cuda_available": False, "error": str(exc), "train_device": TRAIN_DEVICE}


def yolo_label_name(class_id):
    names = model.names if hasattr(model, "names") else {}
    return names.get(int(class_id), str(int(class_id))) if isinstance(names, dict) else names[int(class_id)]


def prediction_for_task(task):
    image_value = task.get("data", {}).get(state["value"])
    image_path, temporary = resolve_image_path(image_value)
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]

        with lock:
            result = model.predict(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        items = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = yolo_label_name(box.cls[0].item())
            if state["labels"] and label not in state["labels"]:
                continue
            items.append({
                "id": str(uuid.uuid4()),
                "from_name": state["from_name"] or "label",
                "to_name": state["to_name"] or state["value"],
                "type": "rectanglelabels",
                "value": {
                    "x": x1 / width * 100,
                    "y": y1 / height * 100,
                    "width": (x2 - x1) / width * 100,
                    "height": (y2 - y1) / height * 100,
                    "rectanglelabels": [label],
                },
                "score": float(box.conf[0].item()),
            })

        score = sum(item["score"] for item in items) / len(items) if items else 0.0
        return {"result": items, "score": score, "model_version": Path(state["model_path"]).name}
    finally:
        if temporary:
            Path(image_path).unlink(missing_ok=True)


def ls_get(path):
    response = requests.get(f"{LABEL_STUDIO_URL}{path}", headers=auth_headers(), timeout=120)
    response.raise_for_status()
    return response.json()


def export_project(project_id):
    response = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/export",
        params={"exportType": "JSON"},
        headers=auth_headers(),
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def annotation_results(task):
    annotations = task.get("annotations") or []
    for annotation in annotations:
        if annotation.get("was_cancelled"):
            continue
        result = annotation.get("result") or []
        if result:
            return result
    return []


def convert_to_yolo_dataset(project_id, tasks):
    run_dir = TRAIN_DIR / f"project-{project_id}-{int(time.time())}"
    images_dir = run_dir / "images" / "train"
    labels_dir = run_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    names = list(state["labels"] or [str(name) for name in model.names.values()])
    if not names:
        raise ValueError("No labels found in Label Studio config or YOLO model")

    count = 0
    for task in tasks:
        results = annotation_results(task)
        if not results:
            continue
        image_value = task.get("data", {}).get(state["value"])
        image_path, temporary = resolve_image_path(image_value)
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            target_image = images_dir / f"{task.get('id', uuid.uuid4())}{Path(image_path).suffix or '.jpg'}"
            shutil.copyfile(image_path, target_image)
            lines = []
            for item in results:
                value = item.get("value", {})
                labels = value.get("rectanglelabels") or []
                if not labels or labels[0] not in names:
                    continue
                class_id = names.index(labels[0])
                x = float(value["x"]) / 100
                y = float(value["y"]) / 100
                w = float(value["width"]) / 100
                h = float(value["height"]) / 100
                lines.append(f"{class_id} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
            if lines:
                (labels_dir / f"{target_image.stem}.txt").write_text("\n".join(lines) + "\n")
                count += 1
        finally:
            if temporary:
                Path(image_path).unlink(missing_ok=True)

    if count == 0:
        raise ValueError("No annotated rectangle labels found to train")

    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(run_dir), "train": "images/train", "val": "images/train", "names": names}))
    return data_yaml, count


def public_job(job):
    public = {key: value for key, value in job.items() if key != "thread"}
    if public.get("run_dir"):
        public["metrics"] = yolo_metrics(public["run_dir"])
    public["elapsed_seconds"] = int((public.get("finished_at") or time.time()) - public.get("started_at", public.get("created_at", time.time())))
    epochs = int((public.get("train_config") or {}).get("epochs") or 0)
    latest = (public.get("metrics") or {}).get("latest") or {}
    current_epoch = metric_float(latest, "epoch")
    public["progress"] = {
        "current_epoch": int(current_epoch) + 1 if current_epoch is not None else 0,
        "total_epochs": epochs,
    }
    if public.get("status") == "running" and public.get("metrics", {}).get("rows"):
        public["phase"] = "training"
        public["message"] = f"Entrenando epoca {public['progress']['current_epoch']}/{epochs}"
    if public.get("status") == "running" and public.get("run_dir") and not public.get("metrics", {}).get("rows"):
        public["message"] = public.get("message") or "YOLO ya inicio, esperando que termine la primera epoca para escribir results.csv."
    return public


def format_duration(seconds):
    seconds = max(int(seconds or 0), 0)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days:
        parts.append(f"{days}D")
    if days or hours:
        parts.append(f"{hours}H")
    if days or hours or minutes:
        parts.append(f"{minutes}M")
    parts.append(f"{seconds}S")
    return " ".join(parts)


def model_metadata_path(model_path):
    return Path(str(model_path) + ".json")


def write_model_metadata(model_path, job):
    metadata = public_job(job)
    metadata["saved_at"] = time.time()
    model_metadata_path(model_path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def remove_path(path):
    if not path:
        return
    target = Path(path)
    if not target.exists():
        return
    allowed_roots = [DATA_DIR.resolve()]
    resolved = target.resolve()
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"Refusing to delete path outside data dir: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def delete_job_artifacts(job):
    trained_model = job.get("trained_model")
    if trained_model:
        remove_path(trained_model)
        remove_path(str(model_metadata_path(trained_model)))
    remove_path(job.get("run_dir"))
    dataset_yaml = job.get("dataset_yaml")
    if dataset_yaml:
        remove_path(Path(dataset_yaml).parent)


def read_model_metadata(model_path):
    path = model_metadata_path(model_path)
    if not path.exists():
        return inferred_model_metadata(model_path)
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"error": f"Could not read metadata: {exc}"}


def inferred_model_metadata(model_path):
    name = Path(model_path).name
    if not name.startswith("project-") or not name.endswith("-best.pt"):
        return None
    stem = name.removesuffix("-best.pt")
    project_part = stem.removeprefix("project-")
    parts = project_part.split("-", 1)
    project_id = parts[0]
    run_dir = RUNS_DIR / stem
    if not run_dir.exists():
        run_dir = RUNS_DIR / f"project-{project_id}"
    if not run_dir.exists():
        return None
    return {
        "id": f"model-{Path(model_path).stem}",
        "status": "completed",
        "phase": "loaded_from_saved_run",
        "message": "Métricas recuperadas desde el directorio de entrenamiento guardado",
        "project": project_id,
        "trained_model": str(model_path),
        "run_dir": str(run_dir),
        "metrics": yolo_metrics(run_dir),
    }


def yolo_metrics(run_dir):
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        return {"rows": [], "latest": {}, "summary": training_artifact_summary(run_dir, [])}
    with path.open(newline="") as file:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(file)]
    return {"rows": rows, "latest": rows[-1] if rows else {}, "summary": training_artifact_summary(run_dir, rows)}


def metric_float(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def training_artifact_summary(run_dir, rows):
    run_path = Path(run_dir)
    weights_dir = run_path / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    best_metric_key = "metrics/mAP50-95(B)"
    best_index = None
    best_metric = None
    for index, row in enumerate(rows):
        value = metric_float(row, best_metric_key)
        if value is not None and (best_metric is None or value > best_metric):
            best_metric = value
            best_index = index
    current_epoch = int(metric_float(rows[-1], "epoch") or len(rows) - 1) if rows else None
    best_epoch = int(metric_float(rows[best_index], "epoch") or best_index) if best_index is not None else None
    epochs_without_improvement = (current_epoch - best_epoch) if current_epoch is not None and best_epoch is not None else None
    patience_remaining = (TRAIN_PATIENCE - epochs_without_improvement) if epochs_without_improvement is not None else None
    return {
        "best_exists": best.exists(),
        "best_path": str(best) if best.exists() else "",
        "best_size": best.stat().st_size if best.exists() else 0,
        "best_modified_at": best.stat().st_mtime if best.exists() else None,
        "last_exists": last.exists(),
        "last_path": str(last) if last.exists() else "",
        "last_size": last.stat().st_size if last.exists() else 0,
        "last_modified_at": last.stat().st_mtime if last.exists() else None,
        "best_metric_key": best_metric_key,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "current_epoch": current_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "patience": TRAIN_PATIENCE,
        "patience_remaining": max(patience_remaining, 0) if patience_remaining is not None else None,
    }


def chart_points(rows, key, width=560, height=180, pad=26):
    values = [metric_float(row, key) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return "", None, None
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = pad + index * ((width - pad * 2) / (len(values) - 1))
        y = height - pad - ((value - min_value) / span) * (height - pad * 2)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points), min_value, max_value


def svg_chart(rows, key, title, color):
    points, min_value, max_value = chart_points(rows, key)
    if not points:
        return f"<div class='chart empty'>Sin datos para {html.escape(title)}</div>"
    latest = metric_float(rows[-1], key)
    help_text = chart_help_text(key)
    return f"""
    <div class='chart'>
      <div class='chart-head'><strong>{html.escape(title)} <span class='help' data-tip='{html.escape(help_text)}'>?</span></strong><span>último: {latest:.4f}</span></div>
      <svg viewBox='0 0 560 180' role='img'>
        <line x1='26' y1='154' x2='534' y2='154' class='axis'/>
        <line x1='26' y1='26' x2='26' y2='154' class='axis'/>
        <polyline points='{points}' fill='none' stroke='{color}' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/>
      </svg>
      <div class='chart-scale'><span>min {min_value:.4f}</span><span>max {max_value:.4f}</span></div>
    </div>
    """


def chart_help_text(key):
    texts = {
        "metrics/mAP50-95(B)": "mAP50-95\nMide la calidad promedio de detección con criterios estrictos de solapamiento.\nMás alto es mejor.\n0.50 es aceptable, 0.70+ suele ser bueno, 0.90+ es excelente si el dataset es representativo.",
        "metrics/mAP50(B)": "mAP50\nMide detecciones correctas con un criterio de solapamiento más permisivo.\nMás alto es mejor.\nSirve para ver si el modelo encuentra los objetos, pero puede ser optimista frente a mAP50-95.",
        "train/box_loss": "Train box loss\nError de localización de cajas en el conjunto de entrenamiento.\nMás bajo es mejor.\nDebe tender a bajar; si baja mucho y la validación empeora puede haber sobreajuste.",
        "val/box_loss": "Val box loss\nError de localización de cajas en validación.\nMás bajo es mejor.\nEs más importante que train loss para saber si generaliza. Si sube mientras train baja, puede haber sobreajuste.",
    }
    return texts.get(key, "Métrica de entrenamiento YOLO. Revisa si mejora de forma estable durante las épocas.")


load_jobs()


def job_card_html(job):
    cancel = f"<button type='button' class='cancel-job' data-cancel-job='{html.escape(job['id'])}'>Cancelar</button>" if job.get("status") == "running" else ""
    best_epoch = (job.get("metrics") or {}).get("summary", {}).get("best_epoch")
    best_text = f"Best época {best_epoch}" if best_epoch is not None else "Best n/d"
    return f"""
    <div class='job-card' data-job='{html.escape(job['id'])}'>
      <span class='status {html.escape(job['status'])}'>{html.escape(job['status'])}</span>
      <strong>Proyecto {html.escape(str(job['project']))}</strong>
      <span class='duration' data-start='{job.get('started_at', job.get('created_at', 0))}' data-finished='{job.get('finished_at', '')}'>{html.escape(format_duration(job.get('elapsed_seconds', 0)))}</span>
      <span class='epoch' data-job-epoch='{html.escape(job['id'])}'>Época {job.get('progress', {}).get('current_epoch', 0)}/{job.get('progress', {}).get('total_epochs', 0)}</span>
      <span class='best-epoch' data-job-best='{html.escape(job['id'])}'>{html.escape(best_text)}</span>
      <small>{html.escape(job['id'][:8])}</small>
      {cancel}
      <button type='button' class='delete-job' data-delete-job='{html.escape(job['id'])}'>Eliminar</button>
    </div>
    """


def model_files():
    models_dir = DATA_DIR / "models"
    if not models_dir.exists():
        return []
    files = []
    for path in sorted(models_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        files.append({
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
            "active": str(path) == state["model_path"],
            "metadata": read_model_metadata(path),
        })
    return files


def available_model_options():
    options = []
    seen = set()
    for path in [Path(MODEL_PATH), Path(TRAIN_MODEL_PATH)]:
        if str(path) not in seen:
            options.append({"name": path.name, "path": str(path), "source": "inicial", "active": str(path) == state["model_path"]})
            seen.add(str(path))
    for item in model_files():
        if item["path"] not in seen:
            options.append({"name": item["name"], "path": item["path"], "source": "entrenado", "active": item["active"]})
            seen.add(item["path"])
    return options


def model_option_label(item):
    path = Path(item["path"])
    metadata = read_model_metadata(path)
    trained_at = None
    map5095 = None
    if metadata:
        trained_at = metadata.get("finished_at") or metadata.get("saved_at") or metadata.get("created_at")
        map5095 = (metadata.get("metrics", {}).get("summary", {}) or {}).get("best_metric")
    if trained_at is None and path.exists():
        trained_at = path.stat().st_mtime
    date = time.strftime("%Y-%m-%d %H:%M", time.localtime(trained_at)) if trained_at else "sin fecha"
    metric = f"mAP50-95 {map5095:.4f}" if isinstance(map5095, (int, float)) else "mAP50-95 n/d"
    return f"{item['name']} · {item['source']} · {date} · {metric}"


def train_async(job_id, project_id, config):
    state["training"] = True
    state["last_train_status"] = {"status": "running", "project": project_id}
    jobs[job_id].update({"status": "running", "phase": "starting", "message": "Inicializando entrenamiento", "started_at": time.time()})
    save_jobs()
    try:
        jobs[job_id].update({"phase": "loading_project", "message": "Leyendo configuracion del proyecto en Label Studio"})
        save_jobs()
        project = ls_get(f"/api/projects/{project_id}")
        parse_label_config(project.get("label_config"))
        jobs[job_id].update({"phase": "exporting", "message": "Exportando tareas y anotaciones desde Label Studio"})
        save_jobs()
        tasks = export_project(project_id)
        jobs[job_id].update({"label_config": project.get("label_config"), "task_count": len(tasks)})
        jobs[job_id].update({"phase": "converting_dataset", "message": "Convirtiendo anotaciones a formato YOLO"})
        save_jobs()
        data_yaml, image_count = convert_to_yolo_dataset(project_id, tasks)
        jobs[job_id].update({"dataset_yaml": str(data_yaml), "images": image_count})
        run_name = f"project-{project_id}-{job_id[:8]}"
        run_dir = RUNS_DIR / run_name
        jobs[job_id]["run_dir"] = str(run_dir)
        jobs[job_id].update({"phase": "training_waiting_first_epoch", "message": "YOLO iniciado; esperando primera epoca para disponer de metricas"})
        save_jobs()
        training_model = YOLO(config["model_path"])
        train_kwargs = {
            "data": str(data_yaml),
            "epochs": config["epochs"],
            "imgsz": config["imgsz"],
            "batch": config["batch"],
            "patience": config["patience"],
            "workers": config["workers"],
            "project": str(RUNS_DIR),
            "name": run_name,
            "exist_ok": False,
        }
        device = resolved_train_device(config)
        if device is not None:
            train_kwargs["device"] = device
        result = training_model.train(**train_kwargs)
        best = Path(result.save_dir) / "weights" / "best.pt"
        if best.exists():
            trained_model = DATA_DIR / "models" / f"project-{project_id}-{job_id[:8]}-best.pt"
            trained_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best, trained_model)
            load_model(str(trained_model))
            jobs[job_id]["trained_model"] = str(trained_model)
        jobs[job_id].update({"status": "completed", "phase": "completed", "message": "Entrenamiento completado", "finished_at": time.time(), "run_dir": str(result.save_dir), "active_model_path": state["model_path"]})
        if jobs[job_id].get("trained_model"):
            write_model_metadata(jobs[job_id]["trained_model"], jobs[job_id])
        save_jobs()
        state["last_train_status"] = {"status": "completed", "project": project_id, "images": image_count, "model_path": state["model_path"], "train_config": config, "job_id": job_id}
    except Exception as exc:
        jobs[job_id].update({"status": "failed", "phase": "failed", "message": "El entrenamiento fallo", "finished_at": time.time(), "error": str(exc)})
        save_jobs()
        state["last_train_status"] = {"status": "failed", "project": project_id, "error": str(exc), "train_config": config, "job_id": job_id}
    finally:
        state["training"] = False


@app.get("/health")
def health():
    return jsonify({"status": "UP", "model_path": state["model_path"], "training": state["training"]})


@app.post("/setup")
def setup():
    payload = request.get_json(force=True, silent=True) or {}
    parse_label_config(payload.get("label_config"))
    return jsonify({"model_version": Path(state["model_path"]).name})


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    tasks = payload.get("tasks", [])
    if payload.get("label_config"):
        parse_label_config(payload.get("label_config"))
    app.logger.info("Predict requested for %s task(s)", len(tasks))
    predictions = [prediction_for_task(task) for task in tasks]
    app.logger.info("Predict completed with %s result(s)", sum(len(item.get("result", [])) for item in predictions))
    return jsonify({"results": predictions, "model_version": Path(state["model_path"]).name})


@app.post("/train")
def train():
    payload = request.get_json(force=True, silent=True) or {}
    project_id = payload.get("project") or payload.get("project_id") or payload.get("project_id_from_task")
    if isinstance(project_id, dict):
        project_id = project_id.get("id")
    if not project_id and payload.get("task"):
        project_id = payload["task"].get("project")
    if not project_id:
        return jsonify({"status": "error", "message": "project id not found in payload"}), 400
    config = request_train_config(payload)
    if state["training"]:
        return jsonify({"status": "busy", "last_train_status": state["last_train_status"], "requested_train_config": config}), 202
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "project": project_id,
        "created_at": time.time(),
        "train_config": config,
    }
    save_jobs()
    thread = threading.Thread(target=train_async, args=(job_id, project_id, config), daemon=True)
    jobs[job_id]["thread"] = thread
    thread.start()
    return jsonify({"status": "queued", "project": project_id, "train_config": config, "job_id": job_id})


@app.get("/api/jobs")
def api_jobs():
    return jsonify({"jobs": [public_job(job) for job in sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True)]})


@app.get("/api/jobs-fragment")
def api_jobs_fragment():
    latest_jobs = [public_job(job) for job in sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True)]
    return "".join(job_card_html(job) for job in latest_jobs) or "<div class='empty'>Todavia no hay jobs de entrenamiento.</div>"


@app.get("/api/jobs/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    return jsonify(public_job(job))


@app.delete("/api/jobs/<job_id>")
def api_delete_job(job_id):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    if job.get("status") == "running":
        return jsonify({"status": "error", "message": "No se puede eliminar un job en ejecucion"}), 409
    try:
        delete_job_artifacts(job)
        jobs.pop(job_id, None)
        save_jobs()
        return jsonify({"status": "deleted", "job_id": job_id})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.post("/api/jobs/<job_id>/cancel")
def api_cancel_job(job_id):
    job = jobs.get(job_id)
    if not job:
        abort(404)
    if job.get("status") != "running":
        return jsonify({"status": "error", "message": "Solo se pueden cancelar jobs en ejecucion"}), 409
    job.update({
        "status": "cancelled",
        "phase": "cancelled",
        "message": "Entrenamiento cancelado por el usuario; reiniciando backend para detener Ultralytics",
        "finished_at": time.time(),
    })
    state["training"] = False
    state["last_train_status"] = {"status": "cancelled", "project": job.get("project"), "job_id": job_id}
    save_jobs()
    threading.Timer(0.5, lambda: _os._exit(0)).start()
    return jsonify({"status": "cancelled", "job_id": job_id, "message": "Backend restarting to stop training"})


@app.get("/api/models")
def api_models():
    return jsonify({"models": model_files(), "available_models": available_model_options(), "active_model_path": state["model_path"]})


@app.post("/api/active-model")
def api_active_model():
    payload = request.get_json(force=True, silent=True) or {}
    path = payload.get("model_path")
    if not path:
        return jsonify({"status": "error", "message": "model_path is required"}), 400
    model_path = Path(path)
    allowed = {item["path"] for item in available_model_options()}
    if str(model_path) not in allowed or not model_path.exists():
        return jsonify({"status": "error", "message": "model_path is not available"}), 400
    load_model(str(model_path))
    return jsonify({"status": "ok", "active_model_path": state["model_path"]})


@app.get("/download/models/<name>")
def download_model(name):
    if "/" in name or ".." in name or not name.endswith(".pt"):
        abort(400)
    path = DATA_DIR / "models" / name
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=name)


@app.get("/status")
def status():
    return jsonify({**state, "gpu": gpu_status(), "train_config": {
        "train_model_path": TRAIN_MODEL_PATH,
        "epochs": TRAIN_EPOCHS,
        "imgsz": TRAIN_IMGSZ,
        "batch": TRAIN_BATCH,
        "patience": TRAIN_PATIENCE,
        "workers": TRAIN_WORKERS,
        "device": TRAIN_DEVICE,
    }})


@app.get("/")
def index():
    latest_jobs = [public_job(job) for job in sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True)]
    selected_job = latest_jobs[0] if latest_jobs else None
    job_cards = "".join(job_card_html(job) for job in latest_jobs) or "<div class='empty'>Todavia no hay jobs de entrenamiento.</div>"
    model_rows = "".join([
        f"""
        <tr>
          <td>{html.escape(model['name'])}{' <span class="pill">activo</span>' if model['active'] else ''}</td>
          <td>{model['size'] / 1024 / 1024:.1f} MB</td>
          <td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model['modified_at']))}</td>
          <td class='row-actions'><button type='button' class='mini' data-model='{html.escape(model['name'])}'>Ver métricas</button><a class='download' href='/download/models/{html.escape(model['name'])}'>Descargar</a></td>
        </tr>
        """
        for model in model_files()
    ]) or "<tr><td colspan='4' class='empty'>Todavia no hay modelos entrenados.</td></tr>"
    selected_json = html.escape(json.dumps(selected_job or {}, indent=2, ensure_ascii=False))
    latest_metrics = (selected_job or {}).get("metrics", {}).get("latest", {}) if selected_job else {}
    selected_metric_rows = (selected_job or {}).get("metrics", {}).get("rows", []) if selected_job else []
    selected_summary = (selected_job or {}).get("metrics", {}).get("summary", {}) if selected_job else {}
    metric_labels = [
        ("epoch", "Epoch"),
        ("train/box_loss", "Train box loss"),
        ("train/cls_loss", "Train cls loss"),
        ("train/dfl_loss", "Train dfl loss"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("val/box_loss", "Val box loss"),
        ("val/cls_loss", "Val cls loss"),
        ("val/dfl_loss", "Val dfl loss"),
        ("lr/pg0", "LR pg0"),
        ("lr/pg1", "LR pg1"),
        ("lr/pg2", "LR pg2"),
    ]
    metric_rows = "".join(
        f"<div class='metric-cell'><small>{html.escape(label)}</small><strong>{html.escape(str(latest_metrics.get(key, '')))}</strong></div>"
        for key, label in metric_labels
    ) if latest_metrics else "<div class='empty'>Sin metricas todavia. Aparecen cuando Ultralytics escribe results.csv.</div>"
    charts_html = "".join([
        svg_chart(selected_metric_rows, "metrics/mAP50-95(B)", "mAP50-95", "#70e000"),
        svg_chart(selected_metric_rows, "metrics/mAP50(B)", "mAP50", "#4cc9f0"),
        svg_chart(selected_metric_rows, "train/box_loss", "Train box loss", "#ffd166"),
        svg_chart(selected_metric_rows, "val/box_loss", "Val box loss", "#ff5d73"),
    ]) if selected_metric_rows else "<div class='empty'>Los gráficos aparecen cuando exista results.csv para el job seleccionado.</div>"
    best_info_rows = "".join([
        f"<tr><td>Best existe</td><td>{selected_summary.get('best_exists', False)}</td></tr>",
        f"<tr><td>Best path</td><td>{html.escape(str(selected_summary.get('best_path', '')))}</td></tr>",
        f"<tr><td>Best mAP50-95</td><td>{html.escape(str(selected_summary.get('best_metric', '')))}</td></tr>",
        f"<tr><td>Best epoch</td><td>{html.escape(str(selected_summary.get('best_epoch', '')))}</td></tr>",
        f"<tr><td>Epoch actual</td><td>{html.escape(str(selected_summary.get('current_epoch', '')))}</td></tr>",
        f"<tr><td>Epochs sin mejora</td><td>{html.escape(str(selected_summary.get('epochs_without_improvement', '')))}</td></tr>",
        f"<tr><td>Paciencia restante</td><td>{html.escape(str(selected_summary.get('patience_remaining', '')))}</td></tr>",
    ]) if selected_summary else "<tr><td colspan='2' class='empty'>Sin información de best/paciencia todavía.</td></tr>"
    selected_message = html.escape(str((selected_job or {}).get("message", "Selecciona un job para ver su estado.")))
    selected_phase = html.escape(str((selected_job or {}).get("phase", (selected_job or {}).get("status", "idle"))))
    initial_selected_job_id = json.dumps((selected_job or {}).get("id"))
    initial_selected_job_status = json.dumps((selected_job or {}).get("status"))
    devices = gpu_status().get("cuda_devices", [])
    device_options = "<option value='auto'>auto</option><option value='cpu'>cpu</option>" + "".join(
        f"<option value='{index}'>{index} - {html.escape(name)}</option>" for index, name in enumerate(devices)
    )
    model_options = "".join(
        f"<option value='{html.escape(item['path'])}' {'selected' if item['path'] == TRAIN_MODEL_PATH else ''}>{html.escape(model_option_label(item))}</option>"
        for item in available_model_options()
    )
    active_model_options = "".join(
        f"<option value='{html.escape(item['path'])}' {'selected' if item['active'] else ''}>{html.escape(model_option_label(item))}</option>"
        for item in available_model_options()
    )
    return f"""
<!doctype html>
<html lang='es'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>YOLO 11 Backend</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0d1321; --panel:#141c2f; --panel2:#19243a; --text:#edf2ff; --muted:#98a6c7; --accent:#70e000; --warn:#ffd166; --bad:#ff5d73; --line:#27344f; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: radial-gradient(circle at top left, #1a2c46, var(--bg) 42%); color:var(--text); }}
    header {{ padding:32px clamp(18px,4vw,48px) 18px; display:flex; justify-content:space-between; gap:18px; align-items:flex-end; }}
    h1 {{ margin:0; font-size:clamp(28px,4vw,44px); letter-spacing:-0.04em; }}
    .subtitle {{ color:var(--muted); margin-top:8px; }}
    .actions {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .btn,.download {{ color:#07110b; background:var(--accent); text-decoration:none; border:0; border-radius:999px; padding:10px 14px; font-weight:750; }}
    main {{ padding:18px clamp(18px,4vw,48px) 42px; display:grid; grid-template-columns: 360px 1fr; gap:18px; }}
    section {{ background:linear-gradient(180deg,var(--panel),#111827); border:1px solid var(--line); border-radius:22px; padding:18px; box-shadow: 0 20px 60px rgba(0,0,0,.28); }}
    h2 {{ margin:0 0 14px; font-size:18px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:18px; }}
    .metric {{ background:var(--panel2); border:1px solid var(--line); border-radius:16px; padding:14px; }}
    .metric small {{ color:var(--muted); display:block; margin-bottom:6px; }}
    .metric strong {{ font-size:18px; overflow-wrap:anywhere; }}
    .metrics-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .metric-cell {{ background:#091120; border:1px solid var(--line); border-radius:14px; padding:12px; min-height:76px; }}
    .metric-cell small {{ color:var(--muted); display:block; margin-bottom:7px; font-size:12px; }}
    .metric-cell strong {{ font-size:15px; overflow-wrap:anywhere; }}
    details {{ border:1px solid var(--line); border-radius:16px; background:#070b13; overflow:hidden; }}
    summary {{ cursor:pointer; padding:14px 16px; color:var(--text); font-weight:800; }}
    details pre {{ border:0; border-top:1px solid var(--line); border-radius:0; }}
    .job-list {{ display:flex; flex-direction:column; gap:10px; }}
    .job-card {{ width:100%; text-align:left; border:1px solid var(--line); color:var(--text); background:#10192b; border-radius:16px; padding:14px; cursor:pointer; display:grid; gap:6px; }}
    .job-card:hover {{ border-color:var(--accent); transform:translateY(-1px); }}
    .status,.pill {{ display:inline-flex; width:max-content; border-radius:999px; padding:4px 9px; background:#24324f; color:var(--muted); font-size:12px; font-weight:800; text-transform:uppercase; }}
    .status.running {{ color:#07110b; background:var(--warn); }} .status.completed {{ color:#07110b; background:var(--accent); }} .status.failed {{ color:white; background:var(--bad); }}
    .duration {{ color:var(--warn); font-weight:800; }}
    .epoch {{ color:#4cc9f0; font-weight:800; }}
    .best-epoch {{ color:var(--accent); font-weight:800; }}
    .row-actions {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    .mini {{ color:var(--text); background:#24324f; border:1px solid var(--line); border-radius:999px; padding:9px 12px; cursor:pointer; font-weight:750; }}
    .delete-job {{ justify-self:start; color:white; background:rgba(255,93,115,.8); border-radius:999px; padding:7px 10px; font-size:12px; font-weight:850; }}
    .cancel-job {{ justify-self:start; color:#07110b; background:var(--warn); border-radius:999px; padding:7px 10px; font-size:12px; font-weight:850; }}
    pre {{ margin:0; white-space:pre-wrap; overflow:auto; max-height:520px; background:#070b13; border:1px solid var(--line); border-radius:16px; padding:16px; color:#dce7ff; }}
    table {{ width:100%; border-collapse:collapse; overflow:hidden; border-radius:16px; }}
    th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:12px; color:var(--text); }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .empty {{ color:var(--muted); padding:14px; }}
    form {{ display:grid; gap:12px; }}
    .form-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
    label {{ display:grid; gap:6px; color:var(--muted); font-size:13px; font-weight:700; }}
    input,select {{ width:100%; border:1px solid var(--line); border-radius:12px; background:#090f1d; color:var(--text); padding:11px 12px; font:inherit; }}
    input:focus,select:focus {{ outline:2px solid rgba(112,224,0,.35); border-color:var(--accent); }}
    .notice {{ margin-top:10px; color:var(--muted); min-height:22px; }}
    .job-message {{ margin-bottom:14px; border:1px solid var(--line); background:#091120; border-radius:16px; padding:14px; display:flex; justify-content:space-between; gap:12px; align-items:center; }}
    .job-message span {{ color:var(--text); }} .job-message strong {{ color:var(--accent); text-transform:uppercase; font-size:12px; }}
    .error-box {{ display:none; margin-bottom:14px; border:1px solid rgba(255,93,115,.7); background:rgba(255,93,115,.12); color:#ffd7dd; border-radius:16px; padding:14px; }}
    .charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .chart {{ background:#070b13; border:1px solid var(--line); border-radius:16px; padding:12px; min-height:230px; }}
    .chart-head,.chart-scale {{ display:flex; justify-content:space-between; gap:10px; color:var(--muted); font-size:12px; }}
    .chart-head strong {{ color:var(--text); font-size:14px; }}
    .help {{ position:relative; display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; margin-left:6px; border-radius:50%; background:#24324f; color:var(--accent); font-size:12px; cursor:help; }}
    .help:hover::after {{ content:attr(data-tip); white-space:pre-line; position:absolute; z-index:20; left:0; top:24px; width:300px; padding:12px; border-radius:12px; background:#050914; color:var(--text); border:1px solid var(--line); box-shadow:0 16px 40px rgba(0,0,0,.45); line-height:1.35; font-weight:500; }}
    .chart svg {{ width:100%; height:180px; display:block; }}
    .axis {{ stroke:#2b3958; stroke-width:1; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns:1fr; }} header {{ align-items:flex-start; flex-direction:column; }} .metric-grid,.metrics-grid {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width: 700px) {{ .form-grid,.charts,.metrics-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>YOLO 11 Backend</h1><div class='subtitle'>Inferencia, entrenamiento y modelos para Label Studio · Ultralytics {html.escape(ULTRALYTICS_VERSION)}</div></div>
    <div class='actions'><a class='btn' href='/'>Refrescar</a><a class='btn' href='/status'>JSON status</a></div>
  </header>
  <main>
    <section>
      <h2>Jobs</h2>
      <div class='job-list' id='job-list'>{job_cards}</div>
    </section>
    <div>
      <section style='margin-bottom:18px'>
        <h2>Lanzar entrenamiento</h2>
        <form id='train-form'>
          <div class='form-grid'>
            <label>Proyecto Label Studio<input name='project' type='number' min='1' value='19' required></label>
            <label>Modelo base<select name='model_path'>{model_options}</select></label>
            <label>Device<select name='device'>{device_options}</select></label>
            <label>Epochs<input name='epochs' type='number' min='1' value='{TRAIN_EPOCHS}' required></label>
            <label>Image size<input name='imgsz' type='number' min='32' step='32' value='{TRAIN_IMGSZ}' required></label>
            <label>Batch<input name='batch' type='number' min='1' value='{TRAIN_BATCH}' required></label>
            <label>Patience<input name='patience' type='number' min='0' value='{TRAIN_PATIENCE}' required></label>
            <label>Workers<input name='workers' type='number' min='0' value='{TRAIN_WORKERS}' required></label>
          </div>
          <button class='btn' type='submit'>Iniciar entrenamiento</button>
          <div id='train-notice' class='notice'></div>
        </form>
      </section>
      <section style='margin-bottom:18px'>
        <h2>Modelo activo para inferencia</h2>
        <form id='active-model-form'>
          <label>Modelo usado por Label Studio para predecir<select name='model_path'>{active_model_options}</select></label>
          <button class='btn' type='submit'>Usar para inferencia</button>
          <div id='active-model-notice' class='notice'></div>
        </form>
      </section>
      <section>
        <h2>Estado</h2>
        <div class='job-message'><span id='job-message'>{selected_message}</span><strong id='job-phase'>{selected_phase}</strong></div>
        <div id='job-error' class='error-box'></div>
        <div class='metric-grid'>
          <div class='metric'><small>Modelo activo</small><strong>{html.escape(Path(state['model_path']).name)}</strong></div>
          <div class='metric'><small>Entrenando</small><strong>{'si' if state['training'] else 'no'}</strong></div>
          <div class='metric'><small>GPU</small><strong>{html.escape(', '.join(gpu_status().get('cuda_devices', [])) or 'sin CUDA')}</strong></div>
        </div>
        <details>
          <summary>Detalle del job seleccionado</summary>
          <pre id='job-detail'>{selected_json}</pre>
        </details>
      </section>
      <section style='margin-top:18px'>
        <h2>Métricas YOLO del job</h2>
        <div class='metrics-grid' id='metrics-body'>{metric_rows}</div>
      </section>
      <section style='margin-top:18px'>
        <h2>Curvas de entrenamiento</h2>
        <div class='charts' id='charts'>{charts_html}</div>
      </section>
      <section style='margin-top:18px'>
        <h2>Best y paciencia</h2>
        <table><thead><tr><th>Dato</th><th>Valor</th></tr></thead><tbody id='best-body'>{best_info_rows}</tbody></table>
      </section>
      <section style='margin-top:18px'>
        <h2>Modelos entrenados</h2>
        <table><thead><tr><th>Modelo</th><th>Tamano</th><th>Modificado</th><th></th></tr></thead><tbody>{model_rows}</tbody></table>
      </section>
    </div>
  </main>
  <script>
    let selectedJobId = {initial_selected_job_id};
    let selectedJobStatus = {initial_selected_job_status};
    bindJobEvents();
    function bindJobEvents() {{
      document.querySelectorAll('.job-card').forEach(btn => btn.onclick = async () => {{
        selectedJobId = btn.dataset.job;
        await loadJob(selectedJobId);
      }});
      document.querySelectorAll('[data-delete-job]').forEach(btn => btn.onclick = async (event) => {{
      event.preventDefault();
      event.stopPropagation();
      const jobId = btn.dataset.deleteJob;
      if (!confirm('Eliminar este job y todos sus datos asociados? Esto borra modelo entrenado, métricas, run y dataset convertido.')) return;
      const res = await fetch('/api/jobs/' + jobId, {{method: 'DELETE'}});
      const body = await res.json().catch(() => ({{}}));
      if (!res.ok) {{ alert('No se pudo eliminar: ' + (body.message || JSON.stringify(body))); return; }}
      if (selectedJobId === jobId) selectedJobId = null;
      location.reload();
      }});
      document.querySelectorAll('[data-cancel-job]').forEach(btn => btn.onclick = async (event) => {{
      event.preventDefault();
      event.stopPropagation();
      const jobId = btn.dataset.cancelJob;
      if (!confirm('Cancelar este entrenamiento en ejecución? El backend se reiniciará para detener YOLO.')) return;
      const res = await fetch('/api/jobs/' + jobId + '/cancel', {{method: 'POST'}});
      const body = await res.json().catch(() => ({{}}));
      if (!res.ok) {{ alert('No se pudo cancelar: ' + (body.message || JSON.stringify(body))); return; }}
      alert('Entrenamiento cancelado. El backend se reiniciará automáticamente.');
      setTimeout(() => location.reload(), 2500);
      }});
    }}
    async function loadJob(jobId) {{
      const res = await fetch('/api/jobs/' + jobId);
      const job = await res.json();
      selectedJobStatus = job.status;
      document.getElementById('job-detail').textContent = JSON.stringify(job, null, 2);
      document.getElementById('job-message').textContent = job.message || '';
      document.getElementById('job-phase').textContent = job.phase || job.status || '';
      const labels = {json.dumps(metric_labels, ensure_ascii=False)};
      const latest = (job.metrics && job.metrics.latest) || {{}};
      const rows = (job.metrics && job.metrics.rows) || [];
      const summary = (job.metrics && job.metrics.summary) || {{}};
      const epochEl = document.querySelector(`[data-job-epoch='${{job.id}}']`);
      if (epochEl && job.progress) epochEl.textContent = `Época ${{job.progress.current_epoch}}/${{job.progress.total_epochs}}`;
      const bestEl = document.querySelector(`[data-job-best='${{job.id}}']`);
      const bestEpoch = summary.best_epoch;
      if (bestEl) bestEl.textContent = bestEpoch === null || bestEpoch === undefined ? 'Best n/d' : `Best época ${{bestEpoch}}`;
      const errorBox = document.getElementById('job-error');
      if (job.error) {{ errorBox.style.display = 'block'; errorBox.textContent = 'Error: ' + job.error; }}
      else {{ errorBox.style.display = 'none'; errorBox.textContent = ''; }}
      document.getElementById('metrics-body').innerHTML = Object.keys(latest).length
        ? labels.map(([key, label]) => `<div class='metric-cell'><small>${{label}}</small><strong>${{latest[key] || ''}}</strong></div>`).join('')
        : `<div class='empty'>Sin metricas todavia. Aparecen cuando Ultralytics escribe results.csv.</div>`;
      document.getElementById('best-body').innerHTML = Object.keys(summary).length
        ? [
            ['Best existe', summary.best_exists], ['Best path', summary.best_path], ['Best mAP50-95', summary.best_metric],
            ['Best epoch', summary.best_epoch], ['Epoch actual', summary.current_epoch],
            ['Epochs sin mejora', summary.epochs_without_improvement], ['Paciencia restante', summary.patience_remaining]
          ].map(([label, value]) => `<tr><td>${{label}}</td><td>${{value ?? ''}}</td></tr>`).join('')
        : `<tr><td colspan='2' class='empty'>Sin información de best/paciencia todavía.</td></tr>`;
      document.getElementById('charts').innerHTML = renderCharts(rows);
    }}
    setInterval(() => {{
      if (selectedJobId && ['running', 'queued'].includes(selectedJobStatus)) loadJob(selectedJobId);
      updateDurations();
      refreshJobList();
    }}, 1000);
    async function refreshJobList() {{
      const res = await fetch('/api/jobs-fragment');
      if (!res.ok) return;
      document.getElementById('job-list').innerHTML = await res.text();
      bindJobEvents();
      updateDurations();
    }}
    function fmt(seconds) {{
      seconds = Math.max(Math.floor(seconds || 0), 0);
      const d = Math.floor(seconds / 86400); seconds %= 86400;
      const h = Math.floor(seconds / 3600); seconds %= 3600;
      const m = Math.floor(seconds / 60); const s = seconds % 60;
      return `${{d ? d + 'D ' : ''}}${{(d || h) ? h + 'H ' : ''}}${{(d || h || m) ? m + 'M ' : ''}}${{s}}S`;
    }}
    function updateDurations() {{
      const now = Date.now() / 1000;
      document.querySelectorAll('.duration').forEach(el => {{
        const start = Number(el.dataset.start || 0);
        const finished = Number(el.dataset.finished || 0);
        if (start) el.textContent = fmt((finished || now) - start);
      }});
    }}
    function renderCharts(rows) {{
      if (!rows.length) return `<div class='empty'>Los gráficos aparecen cuando exista results.csv para el job seleccionado.</div>`;
      return [
        ['metrics/mAP50-95(B)', 'mAP50-95', '#70e000'], ['metrics/mAP50(B)', 'mAP50', '#4cc9f0'],
        ['train/box_loss', 'Train box loss', '#ffd166'], ['val/box_loss', 'Val box loss', '#ff5d73']
      ].map(([key, title, color]) => chart(rows, key, title, color)).join('');
    }}
    function chart(rows, key, title, color) {{
      const values = rows.map(row => Number(row[key])).filter(Number.isFinite);
      if (values.length < 2) return `<div class='chart empty'>Sin datos para ${{title}}</div>`;
      const min = Math.min(...values), max = Math.max(...values), span = Math.max(max - min, 1e-9);
      const points = values.map((value, i) => {{
        const x = 26 + i * ((560 - 52) / (values.length - 1));
        const y = 154 - ((value - min) / span) * 128;
        return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
      }}).join(' ');
      const tips = {{
        'metrics/mAP50-95(B)': 'mAP50-95\\nMide la calidad promedio de detección con criterios estrictos de solapamiento.\\nMás alto es mejor.\\n0.50 es aceptable, 0.70+ suele ser bueno, 0.90+ es excelente si el dataset es representativo.',
        'metrics/mAP50(B)': 'mAP50\\nMide detecciones correctas con un criterio de solapamiento más permisivo.\\nMás alto es mejor.\\nSirve para ver si el modelo encuentra los objetos, pero puede ser optimista frente a mAP50-95.',
        'train/box_loss': 'Train box loss\\nError de localización de cajas en el conjunto de entrenamiento.\\nMás bajo es mejor.\\nDebe tender a bajar; si baja mucho y la validación empeora puede haber sobreajuste.',
        'val/box_loss': 'Val box loss\\nError de localización de cajas en validación.\\nMás bajo es mejor.\\nEs más importante que train loss para saber si generaliza. Si sube mientras train baja, puede haber sobreajuste.'
      }};
      return `<div class='chart'><div class='chart-head'><strong>${{title}} <span class='help' data-tip='${{tips[key] || ''}}'>?</span></strong><span>último: ${{values.at(-1).toFixed(4)}}</span></div><svg viewBox='0 0 560 180'><line x1='26' y1='154' x2='534' y2='154' class='axis'/><line x1='26' y1='26' x2='26' y2='154' class='axis'/><polyline points='${{points}}' fill='none' stroke='${{color}}' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/></svg><div class='chart-scale'><span>min ${{min.toFixed(4)}}</span><span>max ${{max.toFixed(4)}}</span></div></div>`;
    }}
    document.getElementById('train-form').addEventListener('submit', async (event) => {{
      event.preventDefault();
      const notice = document.getElementById('train-notice');
      const data = Object.fromEntries(new FormData(event.currentTarget).entries());
      data.project = Number(data.project);
      data.epochs = Number(data.epochs);
      data.imgsz = Number(data.imgsz);
      data.batch = Number(data.batch);
      data.patience = Number(data.patience);
      data.workers = Number(data.workers);
      notice.textContent = `Enviando entrenamiento: ${{data.epochs}} epochs, paciencia ${{data.patience}}, batch ${{data.batch}}, workers ${{data.workers}}...`;
      const res = await fetch('/train', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(data)
      }});
      const body = await res.json();
      if (!res.ok) {{
        notice.textContent = 'Error: ' + JSON.stringify(body);
        return;
      }}
      if (body.status === 'busy') {{
        notice.textContent = 'Ya hay un entrenamiento corriendo. Configuración solicitada: ' + JSON.stringify(body.requested_train_config);
        return;
      }}
      notice.textContent = 'Job creado: ' + body.job_id + '. Refrescando...';
      selectedJobId = body.job_id;
      await loadJob(selectedJobId);
      await refreshJobList();
      notice.textContent = 'Job creado y seleccionado: ' + body.job_id;
    }});
    document.getElementById('active-model-form').addEventListener('submit', async (event) => {{
      event.preventDefault();
      const notice = document.getElementById('active-model-notice');
      const data = Object.fromEntries(new FormData(event.currentTarget).entries());
      notice.textContent = 'Cambiando modelo activo...';
      const res = await fetch('/api/active-model', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(data)}});
      const body = await res.json();
      notice.textContent = res.ok ? 'Modelo activo: ' + body.active_model_path : 'Error: ' + JSON.stringify(body);
      if (res.ok) setTimeout(() => location.reload(), 900);
    }});
    document.querySelectorAll('button[data-model]').forEach(btn => btn.addEventListener('click', async () => {{
      const res = await fetch('/api/models');
      const body = await res.json();
      const model = body.models.find(item => item.name === btn.dataset.model);
      if (!model || !model.metadata) {{
        document.getElementById('job-message').textContent = 'Este modelo no tiene métricas guardadas.';
        return;
      }}
      const job = model.metadata;
      selectedJobId = null;
      selectedJobStatus = job.status;
      document.getElementById('job-detail').textContent = JSON.stringify(job, null, 2);
      document.getElementById('job-message').textContent = 'Métricas guardadas para ' + model.name;
      document.getElementById('job-phase').textContent = 'modelo guardado';
      const rows = (job.metrics && job.metrics.rows) || [];
      const latest = (job.metrics && job.metrics.latest) || {{}};
      const summary = (job.metrics && job.metrics.summary) || {{}};
      const labels = {json.dumps(metric_labels, ensure_ascii=False)};
      document.getElementById('metrics-body').innerHTML = Object.keys(latest).length ? labels.map(([key, label]) => `<div class='metric-cell'><small>${{label}}</small><strong>${{latest[key] || ''}}</strong></div>`).join('') : `<div class='empty'>Sin metricas guardadas.</div>`;
      document.getElementById('best-body').innerHTML = Object.keys(summary).length ? [
        ['Best existe', summary.best_exists], ['Best path', summary.best_path], ['Best mAP50-95', summary.best_metric], ['Best epoch', summary.best_epoch], ['Epoch actual', summary.current_epoch], ['Epochs sin mejora', summary.epochs_without_improvement], ['Paciencia restante', summary.patience_remaining]
      ].map(([label, value]) => `<tr><td>${{label}}</td><td>${{value ?? ''}}</td></tr>`).join('') : `<tr><td colspan='2' class='empty'>Sin información guardada.</td></tr>`;
      document.getElementById('charts').innerHTML = renderCharts(rows);
    }}));
    updateDurations();
  </script>
</body>
</html>
"""
