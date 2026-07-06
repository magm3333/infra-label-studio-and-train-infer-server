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

_VERSION_FILE = Path(__file__).parent / "VERSION"
APP_VERSION = _VERSION_FILE.read_text().strip() if _VERSION_FILE.exists() else "dev"

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
TRAIN_SPLIT_PERCENT = int(os.getenv("TRAIN_SPLIT_PERCENT", "70"))
TRAIN_LR0 = float(os.getenv("TRAIN_LR0", "0.01"))
TRAIN_WEIGHT_DECAY = float(os.getenv("TRAIN_WEIGHT_DECAY", "0.0005"))
TRAIN_COS_LR = os.getenv("TRAIN_COS_LR", "false").lower() in ("1", "true", "yes", "on")
TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "auto")

DATA_DIR = Path("/app/data")
RUNS_DIR = DATA_DIR / "runs"
TRAIN_DIR = DATA_DIR / "training"
EXTERNAL_MODELS_DIR = DATA_DIR / "external-models"
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
    state["labels"] = labels  # Vacío = sin filtro de clases en predict


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

    train_percent = int(value("train_percent", TRAIN_SPLIT_PERCENT, "train-percent", "Train-Percent"))
    train_percent = min(max(train_percent, 1), 100)
    cos_lr = value("cos_lr", TRAIN_COS_LR, "cos-lr", "Cos-Lr")
    if isinstance(cos_lr, str):
        cos_lr = cos_lr.lower() in ("1", "true", "yes", "on")
    return {
        "model_path": value("model_path", TRAIN_MODEL_PATH, "model-path", "Model-Path"),
        "epochs": int(value("epochs", TRAIN_EPOCHS, "Epochs")),
        "imgsz": int(value("imgsz", TRAIN_IMGSZ, "image_size", "Image-Size", "Imgsz")),
        "batch": int(value("batch", TRAIN_BATCH, "Batch")),
        "patience": int(value("patience", TRAIN_PATIENCE, "Patience")),
        "workers": int(value("workers", TRAIN_WORKERS, "Workers")),
        "train_percent": train_percent,
        "lr0": float(value("lr0", TRAIN_LR0, "learning_rate", "Learning-Rate", "Lr0")),
        "weight_decay": float(value("weight_decay", TRAIN_WEIGHT_DECAY, "weight-decay", "Weight-Decay")),
        "cos_lr": bool(cos_lr),
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


def label_studio_projects():
    try:
        data = ls_get("/api/projects")
        items = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        projects = []
        for item in items:
            project_id = item.get("id")
            if project_id is None:
                continue
            title = item.get("title") or item.get("name") or f"Proyecto {project_id}"
            task_count = item.get("task_number") or item.get("num_tasks") or item.get("task_count") or 0
            projects.append({"id": project_id, "title": str(title), "task_count": task_count})
        return sorted(projects, key=lambda item: str(item["title"]).lower())
    except Exception as exc:
        app.logger.warning("Could not load Label Studio projects: %s", exc)
        return []


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


def convert_to_yolo_dataset(project_id, tasks, train_percent):
    run_dir = TRAIN_DIR / f"project-{project_id}-{int(time.time())}"
    images_dir = run_dir / "images" / "all"
    labels_dir = run_dir / "labels" / "all"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    names = list(state["labels"] or [str(name) for name in model.names.values()])
    if not names:
        raise ValueError("No labels found in Label Studio config or YOLO model")

    samples = []
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
            if target_image.exists() or target_image.is_symlink():
                target_image.unlink()
            if temporary:
                shutil.copyfile(image_path, target_image)
            else:
                target_image.symlink_to(Path(image_path).resolve())
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
                split_key = f"{project_id}:{task.get('id', '')}:{image_value}:{target_image.name}"
                samples.append((hashlib.sha256(split_key.encode()).hexdigest(), target_image))
        finally:
            if temporary:
                Path(image_path).unlink(missing_ok=True)

    if not samples:
        raise ValueError("No annotated rectangle labels found to train")

    samples.sort(key=lambda item: item[0])
    train_count = round(len(samples) * train_percent / 100)
    train_count = min(max(train_count, 1), len(samples))
    train_images = [path for _, path in samples[:train_count]]
    val_images = [path for _, path in samples[train_count:]]
    train_manifest = run_dir / "train.txt"
    val_manifest = run_dir / "val.txt"
    train_manifest.write_text("\n".join(str(path) for path in train_images) + "\n")
    val_manifest.write_text("\n".join(str(path) for path in val_images) + ("\n" if val_images else ""))

    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(run_dir), "train": str(train_manifest), "val": str(val_manifest), "names": names}))
    dataset_info = {
        "images": len(samples),
        "train_percent": train_percent,
        "val_percent": 100 - train_percent,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "image_storage": "symlink_manifest",
    }
    return data_yaml, dataset_info


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


load_jobs()


def job_card_html(job):
    cancel = f"<button type='button' class='cancel-job' data-cancel-job='{html.escape(job['id'])}'>Cancelar</button>" if job.get("status") == "running" else ""
    summary = (job.get("metrics") or {}).get("summary", {})
    best_epoch = summary.get("best_epoch")
    best_metric = summary.get("best_metric")
    if best_epoch is not None and isinstance(best_metric, (int, float)):
        best_text = f"Best época {best_epoch} · mAP50-95 {best_metric:.4f}"
    elif best_epoch is not None:
        best_text = f"Best época {best_epoch}"
    else:
        best_text = "Best n/d"
    dataset = job.get("dataset") or {}
    if dataset:
        dataset_text = f"Train {dataset.get('train_images', 0)} / Valid {dataset.get('val_images', 0)} ({dataset.get('train_percent', 0)}%)"
    elif job.get("images"):
        dataset_text = f"Imágenes {job.get('images')}"
    else:
        dataset_text = "Dataset pendiente"
    return f"""
    <div class='job-card' data-job='{html.escape(job['id'])}'>
      <span class='status {html.escape(job['status'])}'>{html.escape(job['status'])}</span>
      <strong>Proyecto {html.escape(str(job['project']))}</strong>
      <span>{html.escape(dataset_text)}</span>
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


def external_model_files():
    if not EXTERNAL_MODELS_DIR.exists():
        return []
    files = []
    for path in sorted(EXTERNAL_MODELS_DIR.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        files.append({
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
            "active": str(path) == state["model_path"],
            "source": "externo",
            "project": None,
        })
    return files


def model_project(item):
    metadata = item.get("metadata") or read_model_metadata(item.get("path")) or {}
    project = metadata.get("project")
    if project is not None:
        return str(project)
    name = str(item.get("name") or Path(str(item.get("path", ""))).name)
    if name.startswith("project-"):
        parts = name.removeprefix("project-").split("-", 1)
        if parts and parts[0].isdigit():
            return parts[0]
    return "19"


def model_size_mb(path):
    try:
        return Path(path).stat().st_size / 1024 / 1024
    except OSError:
        return None


def available_model_options():
    options = []
    seen = set()
    for path in [Path(MODEL_PATH), Path(TRAIN_MODEL_PATH)]:
        if str(path) not in seen:
            options.append({"name": path.name, "path": str(path), "source": "externo", "active": str(path) == state["model_path"], "project": None, "size": path.stat().st_size if path.exists() else 0})
            seen.add(str(path))
    for item in external_model_files():
        if item["path"] not in seen:
            options.append(item)
            seen.add(item["path"])
    for item in model_files():
        if item["path"] not in seen:
            options.append({"name": item["name"], "path": item["path"], "source": "entrenado", "active": item["active"], "project": model_project(item), "size": item["size"]})
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
    size = model_size_mb(item["path"])
    size_label = f"{size:.1f} MB" if size is not None else "tamano n/d"
    project = item.get("project")
    project_label = f" · proyecto {project}" if item.get("source") == "entrenado" and project is not None else ""
    return f"{item['name']} · {item['source']}{project_label} · {size_label} · {date} · {metric}"


def model_map5095(model):
    metadata = model.get("metadata") or {}
    value = (metadata.get("metrics", {}).get("summary", {}) or {}).get("best_metric")
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/d"


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
        data_yaml, dataset_info = convert_to_yolo_dataset(project_id, tasks, config["train_percent"])
        image_count = dataset_info["images"]
        jobs[job_id].update({"dataset_yaml": str(data_yaml), "images": image_count, "dataset": dataset_info})
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
            "lr0": config["lr0"],
            "weight_decay": config["weight_decay"],
            "cos_lr": config["cos_lr"],
            "project": str(RUNS_DIR),
            "name": run_name,
            "exist_ok": False,
            "val": dataset_info["val_images"] > 0,
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
        state["last_train_status"] = {"status": "completed", "project": project_id, "images": image_count, "dataset": dataset_info, "model_path": state["model_path"], "train_config": config, "job_id": job_id}
    except Exception as exc:
        jobs[job_id].update({"status": "failed", "phase": "failed", "message": "El entrenamiento fallo", "finished_at": time.time(), "error": str(exc)})
        save_jobs()
        state["last_train_status"] = {"status": "failed", "project": project_id, "error": str(exc), "train_config": config, "job_id": job_id}
    finally:
        state["training"] = False


@app.get("/health")
def health():
    return jsonify({"status": "UP", "version": APP_VERSION, "model_path": state["model_path"], "training": state["training"]})


@app.get("/api/version")
def api_version():
    return jsonify({"version": APP_VERSION})


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


@app.post("/api/external-models")
def api_upload_external_model():
    upload = request.files.get("model")
    if not upload or not upload.filename:
        return jsonify({"status": "error", "message": "model file is required"}), 400
    filename = Path(upload.filename).name
    if not filename.endswith(".pt"):
        return jsonify({"status": "error", "message": "Only .pt model files are supported"}), 400
    import re

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip(".-") or f"model-{int(time.time())}.pt"
    if not safe_name.endswith(".pt"):
        safe_name = f"{safe_name}.pt"
    EXTERNAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = EXTERNAL_MODELS_DIR / safe_name
    if target.exists():
        target = EXTERNAL_MODELS_DIR / f"{target.stem}-{int(time.time())}{target.suffix}"
    upload.save(target)
    return jsonify({"status": "ok", "model": {"name": target.name, "path": str(target), "size": target.stat().st_size, "source": "externo"}})


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
        "train_percent": TRAIN_SPLIT_PERCENT,
        "lr0": TRAIN_LR0,
        "weight_decay": TRAIN_WEIGHT_DECAY,
        "cos_lr": TRAIN_COS_LR,
        "device": TRAIN_DEVICE,
    }})


@app.get("/")
def index():
    initial_data = json.dumps({
        "projects": label_studio_projects(),
        "devices": gpu_status().get("cuda_devices", []),
        "currentModelName": Path(state["model_path"]).name,
        "currentModelPath": state["model_path"],
        "confidenceThreshold": CONFIDENCE_THRESHOLD,
        "ultralytics": ULTRALYTICS_VERSION,
        "appVersion": APP_VERSION,
        "defaults": {
            "epochs": TRAIN_EPOCHS,
            "imgsz": TRAIN_IMGSZ,
            "batch": TRAIN_BATCH,
            "patience": TRAIN_PATIENCE,
            "workers": TRAIN_WORKERS,
            "splitPercent": TRAIN_SPLIT_PERCENT,
            "lr0": TRAIN_LR0,
            "weightDecay": TRAIN_WEIGHT_DECAY,
            "cosLr": TRAIN_COS_LR,
            "device": TRAIN_DEVICE,
        },
    }, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO Server &amp; Trainer</title>
  <link rel="icon" href="data:image/x-icon;base64,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8ak3bOesRaaz50jb2X247ew+1J7xi/+wf/nGPnviT//FtnUttJ09R9iO7sNr/pZXr+Upp5yy76abbloHUCwW1YUXXngYEPT396dvuOGGJVJKm0wm9WOPPfZmKpUq3Hnnnb0PPvjgfCGEnTNnTuGpp576/fbt23/z2muv/e6www4bFULYF198cfbFF1+82HXdEBDpdDqUUloppfU8b5r7u65rtdbO2WefffSuXbvSUkp77rnnbtuyZctL27Zte+nLX/7ye67rmomJCefCCy88ctPmDxpRnjbGRAFVAEZXf5ZjZbRZKuzUmSatoHItgyBIfu1rX/vgyCOPHAZ4/fXXOx544IGeq6+++pDh4eGEMUZccsklW4477rhBrXXqvvvuOzC2GHHXXXetO+uss3Z1dnYGK1euHHj88cf/mEqltJTSPvbYY93btm1LAyYMQ2mMEcYYUX/EtRo4g/vuu2/21q1bG4UQdvny5SOPPPLIWwcddFBh3rx5pdtuu+3dVatW7RJCkM/nnW9+65s9QOhPjInknG66z7sACJGOoPuCL+K0doEQFLZtRpf9GTZQRfW6+uShhx56O5VKaSGEveKKKw5/8sknuwEWL148ds8992wC1Lp169I7d+5stNaKrq6u4qc//ekhY0wytp7kkiVLxpcsWTJmjBH5fN557bXXmgE9lemZ2quvvtoeb1uLVatW7QFssVh0fN+XgHvOOef0VZKgP/zh962ASra2Wp0v0fXJczns+z9j2Q9+Rtfpq8DX6CDPnl89jUo1zLDm1NQYXde1vu87y5YtG7322mvft9YK3/eF1hrP88z999//NvFv1Xbt2pUIgkAAdHR0lGLfR0qJ4zgWoLu7u1CZZs+ePckpNjetxTtIor+/P1kRVE9PTxEQjuNYFUf8efPmlV3XtdZaxsfGPDDO3heetkKGKC9Joqsbt20OeAlIwKY7b6SwdQcqmZzyW8PalExE5wM8z7PGGPfmm2/esmbNmllvvfVWq7WWiy66aMsJJ5wwVCgUEul02q/FCMaYGUvIFXwQC+ajVR+3jxq7zn2sBaTd8+wvePeRRzno7M/hzulGCkthxwf0P/csxa3bcbNNM2CA2qHt5A8mwjAUnufpZcuWja1du7YV4Pjjjx8BKmu2XLBgQTGVSulisaj6+vrSIyMjbktLS2CMETEDavPmzY2VMefPn5+f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" />
  <link href="https://cdn.jsdelivr.net/npm/vuetify@3/dist/vuetify.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/@mdi/font@7/css/materialdesignicons.min.css" rel="stylesheet">
  <style>
    html, body {{ height: 100%; margin: 0; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; }}
    .chart-wrap {{ background: #070b13; border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 12px; }}
    .chart-wrap svg {{ width: 100%; height: 180px; display: block; }}
    .axis {{ stroke: #2b3958; stroke-width: 1; }}
    .v-application {{ background: #121212 !important; }}
    .nv-table {{ width: 100%; border-collapse: collapse; }}
    .nv-table th {{ text-align: left; font-size: 11px; font-weight: 600; color: rgba(255,255,255,.5); padding: 6px 16px; border-bottom: 1px solid rgba(255,255,255,.12); }}
    .nv-table td {{ font-size: 12px; padding: 6px 16px; border-bottom: 1px solid rgba(255,255,255,.06); vertical-align: middle; }}
    .nv-table tbody tr:hover {{ background: rgba(255,255,255,.03); }}
  </style>
</head>
<body>
<div id="app">
  <v-app theme="dark">

    <!-- ════════════════════════════════════ APP BAR ════════════════════════════════════ -->
    <v-app-bar color="surface" elevation="2" height="56">
      <v-app-bar-title>
        <div class="d-flex align-center" style="gap:10px">
          <svg height="20" viewBox="0 0 760 560" xmlns="http://www.w3.org/2000/svg">
            <defs><style>.r{{fill:#e31e27}}</style></defs>
            <path class="r" d="m752.23,0v119.81l-40.82-40.82v-27.38c0-7.24-5.87-13.1-13.1-13.1h-27.38L632.42,0h119.81Z"/>
            <path class="r" d="m682.83,138.25c1.37,0,2.73-.04,2.73-1.59,0-1.27-1.05-1.56-2.16-1.56h-2.05v3.15h1.48Zm-1.48,4.99h-1.12v-9.09h3.48c2.05,0,2.96.89,2.96,2.51s-1.05,2.32-2.28,2.54l2.71,4.03h-1.31l-2.56-4.03h-1.88v4.03Zm-4.9-4.54c0,3.76,2.83,6.74,6.64,6.74s6.64-2.98,6.64-6.74-2.83-6.74-6.64-6.74-6.64,2.98-6.64,6.74m14.49,0c0,4.41-3.42,7.86-7.86,7.86s-7.86-3.44-7.86-7.86,3.42-7.86,7.86-7.86,7.86,3.44,7.86,7.86"/>
            <path class="r" d="m690.61,67.11l-28.89,79h-13.17l-20.44-56.47-20.28,56.47h-13.17l-28.9-79h14.78l20.27,55.12c.28.77,1.37.76,1.65,0l20.12-55.12h11.37l19.81,55.23c.28.77,1.36.78,1.65,0l20.42-55.24h14.78Z"/>
            <path class="r" d="m524.92,135.46c-17.06,0-28.52-11.73-28.52-29.19s11.46-28.69,28.52-28.69,28.53,11.53,28.53,28.69-11.46,29.19-28.53,29.19m0-71.31c-25.38,0-43.11,17.32-43.11,42.13s17.73,42.46,43.11,42.46,43.11-17.46,43.11-42.46-17.73-42.13-43.11-42.13"/>
            <path class="r" d="m456.83,19.74h14.26v126.81h-14.26V19.74Z"/>
            <path class="r" d="m423.28,50.04v16.64c0,.48.39.88.88.88h21.96v13.27h-21.96c-.48,0-.88.39-.88.88v64.85h-14.26v-64.85c0-.48-.39-.88-.88-.88h-16.54v-13.27h16.54c.48,0,.88-.39.88-.88v-16.81c0-18.31,11.25-30.13,28.66-30.13,2.81,0,5.86.3,9.1.89v12.8c-1.99-.25-6.68-.42-8.77-.42-9.08,0-14.73,6.52-14.73,17.02"/>
            <path class="r" d="m346.49,136.24c-16.03,0-26.39-11.46-26.39-29.19s10.17-29.18,27.21-29.18c8.18,0,15.26.53,23.99,1.81.43.06.76.44.76.87v46.71c0,.3-.16.59-.41.75-9.09,5.83-16.46,8.23-25.16,8.23m-.66-71.15c-24.12,0-39.82,16.34-39.99,41.63,0,25.29,15.81,42.29,39.33,42.29,9.58,0,17.75-2.4,26.88-7.95v5.48h14.26v-77.08l-1.25-.24c-16.37-3.17-25.53-4.13-39.23-4.13"/>
            <path class="r" d="m304.92,65.75h1.54v14.09h-1.54c-9.56,0-18.2.24-28.72,1.39-.45.05-.8.43-.8.88v64.45h-14.26v-76.61l1.27-.23c14.75-2.63,29.05-3.97,42.5-3.97"/>
            <path class="r" d="m231.09,67.56h14.26v77.03l-1.19.28c-10.71,2.52-27.17,4.15-41.92,4.15-20.91,0-32.43-11.58-32.43-32.6v-48.87h14.26v46.73c0,14.36,7.02,21.95,20.31,21.95,8.39,0,17.12-.66,25.96-1.97.43-.06.76-.44.76-.88v-65.84Z"/>
            <path class="r" d="m159.41,131.44l2.38,12.34-1.31.41c-9.48,2.98-20.54,4.84-28.89,4.84-25.79,0-43.11-17-43.11-42.29s17.72-41.96,44.1-41.96c8.6,0,18.99,1.64,26.46,4.19l1.26.43-2.18,11.97-1.6-.41c-8.48-2.16-16.97-3.4-23.28-3.4-18.43,0-30.33,11.52-30.33,29.35s11.66,29.18,29.02,29.18c6.17,0,15.85-1.58,25.9-4.22l1.59-.42Z"/>
            <path class="r" d="m39.43,136.24c-16.03,0-26.39-11.46-26.39-29.19s10.17-29.18,27.21-29.18c8.18,0,15.26.53,23.99,1.81.43.06.76.44.76.87v46.71c0,.3-.16.59-.41.75-9.09,5.83-16.46,8.23-25.16,8.23m-.66-71.15C15.87,65.1.17,81.43,0,106.73c0,25.29,15.81,42.29,39.33,42.29,9.58,0,17.75-2.4,26.88-7.95v5.48h14.26v-77.08l-1.25-.24c-16.36-3.17-25.53-4.13-39.23-4.13"/>
          </svg>
          <span style="font-size:1rem;font-weight:700;letter-spacing:-.02em">YOLO Server &amp; Trainer</span>
          <v-chip size="x-small" variant="tonal" color="primary" style="font-size:10px">v{{{{ appVersion }}}}</v-chip>
        </div>
      </v-app-bar-title>
      <template v-slot:append>
        <v-chip v-if="activeModelName" class="mr-2" prepend-icon="mdi-brain" size="small" color="primary" variant="tonal">
          {{{{ activeModelName }}}}
        </v-chip>
        <v-chip class="mr-2" size="small"
                :color="isRunning ? 'warning' : 'success'"
                :prepend-icon="isRunning ? 'mdi-refresh mdi-spin' : 'mdi-check-circle'"
                variant="tonal">
          {{{{ isRunning ? 'Entrenando' : 'Listo' }}}}
        </v-chip>
      </template>
    </v-app-bar>

    <!-- ════════════════════════════════════ MAIN ════════════════════════════════════ -->
    <v-main>
      <v-container fluid class="pa-4">

        <v-tabs v-model="tab" color="primary" class="mb-4">
          <v-tab value="inferencia" prepend-icon="mdi-magnify">Inferencia</v-tab>
          <v-tab value="entrenar"   prepend-icon="mdi-weight-lifter">Entrenar</v-tab>
          <v-tab value="historial"  prepend-icon="mdi-history">Historial</v-tab>
        </v-tabs>

        <v-tabs-window v-model="tab">

          <!-- ═════════════════ TAB: INFERENCIA ═════════════════ -->
          <v-tabs-window-item value="inferencia">
            <v-row>

              <!-- Modelo activo -->
              <v-col cols="12" md="7">
                <v-card variant="outlined" class="mb-4">
                  <v-card-title class="text-body-1 font-weight-bold pt-4 pb-1">
                    <v-icon class="mr-2" color="primary">mdi-brain</v-icon>
                    Modelo activo para inferencia
                  </v-card-title>
                  <v-card-text>
                    <v-select
                      v-model="inferenceModelPath"
                      :items="availableModels"
                      item-title="label"
                      item-value="path"
                      label="Modelo"
                      variant="outlined"
                      density="compact"
                      class="mb-3"
                    ></v-select>
                    <div class="d-flex align-center ga-2 mb-4 flex-wrap">
                      <v-chip color="primary" size="small" prepend-icon="mdi-tune-variant">
                        Confianza: {{{{ (confidence * 100).toFixed(0) }}}}%
                      </v-chip>
                      <v-chip
                        v-for="label in activeLabels" :key="label"
                        color="secondary" size="small" variant="tonal"
                        prepend-icon="mdi-tag-outline"
                      >
                        {{{{ label }}}}
                      </v-chip>
                      <span v-if="activeLabels.length === 0" class="text-caption text-medium-emphasis">
                        Sin etiquetas cargadas — Label Studio aún no hizo setup
                      </span>
                    </div>
                    <v-btn
                      color="primary"
                      variant="flat"
                      :loading="applyingModel"
                      :disabled="!inferenceModelPath"
                      @click="applyInferenceModel"
                      prepend-icon="mdi-check"
                    >
                      Aplicar modelo
                    </v-btn>
                    <v-alert
                      v-if="inferenceMsg"
                      :type="inferenceMsgType"
                      variant="tonal"
                      class="mt-3"
                      density="compact"
                    >
                      {{{{ inferenceMsg }}}}
                    </v-alert>
                  </v-card-text>
                </v-card>
              </v-col>

              <!-- Info de conexión -->
              <v-col cols="12" md="5">
                <v-card variant="outlined" class="mb-4">
                  <v-card-title class="text-body-1 font-weight-bold pt-4 pb-1">
                    <v-icon class="mr-2" color="success">mdi-api</v-icon>
                    Conexión Label Studio
                  </v-card-title>
                  <v-card-text>
                    <table style="width:100%;border-collapse:collapse" class="nv-table">
                      <tbody>
                        <tr>
                          <td class="text-caption text-medium-emphasis" style="width:110px">Predicción</td>
                          <td><code class="text-caption">POST /predict</code></td>
                        </tr>
                        <tr>
                          <td class="text-caption text-medium-emphasis">Setup</td>
                          <td><code class="text-caption">POST /setup</code></td>
                        </tr>
                        <tr>
                          <td class="text-caption text-medium-emphasis">Health</td>
                          <td><code class="text-caption">GET /health</code></td>
                        </tr>
                        <tr>
                          <td class="text-caption text-medium-emphasis">Confianza</td>
                          <td class="text-caption">{{{{ (confidence * 100).toFixed(0) }}}}% (env CONFIDENCE_THRESHOLD)</td>
                        </tr>
                        <tr>
                          <td class="text-caption text-medium-emphasis">Ultralytics</td>
                          <td class="text-caption">{{{{ ultralytics }}}}</td>
                        </tr>
                      </tbody>
                    </table>
                  </v-card-text>
                </v-card>

                <v-card variant="outlined">
                  <v-card-title class="text-body-1 font-weight-bold pt-4 pb-1">
                    <v-icon class="mr-2" color="warning">mdi-folder-multiple</v-icon>
                    Modelos disponibles
                  </v-card-title>
                  <v-card-text class="pa-0">
                    <v-list density="compact" class="pa-0">
                      <v-list-item
                        v-for="m in availableModels" :key="m.path"
                        :subtitle="m.source"
                        density="compact"
                      >
                        <template v-slot:prepend>
                          <v-icon size="16" :color="m.active ? 'primary' : 'grey'">
                            {{{{ m.active ? 'mdi-check-circle' : 'mdi-circle-outline' }}}}
                          </v-icon>
                        </template>
                        <v-list-item-title class="text-caption">{{{{ m.label }}}}</v-list-item-title>
                        <template v-slot:append>
                          <v-chip v-if="m.active" size="x-small" color="primary">activo</v-chip>
                        </template>
                      </v-list-item>
                      <v-list-item v-if="availableModels.length === 0">
                        <v-list-item-title class="text-caption text-medium-emphasis">
                          Sin modelos disponibles
                        </v-list-item-title>
                      </v-list-item>
                    </v-list>
                  </v-card-text>
                </v-card>
              </v-col>

            </v-row>
          </v-tabs-window-item>

          <!-- ═════════════════ TAB: ENTRENAR ═════════════════ -->
          <v-tabs-window-item value="entrenar">
            <v-row>

              <!-- Formulario de entrenamiento -->
              <v-col cols="12" md="6">
                <v-card variant="outlined" class="mb-4">
                  <v-card-title class="text-body-1 font-weight-bold pt-4 pb-1">
                    <v-icon class="mr-2" color="primary">mdi-cog</v-icon>
                    Configuración de entrenamiento
                  </v-card-title>
                  <v-card-text>
                    <v-alert v-if="trainProjects.length === 0" type="warning" variant="tonal" density="compact" class="mb-4">
                      No se pudieron cargar proyectos desde Label Studio. Revisá conectividad y API key.
                    </v-alert>

                    <v-row dense>
                      <v-col cols="12">
                        <field-label
                          label="Proyecto Label Studio"
                          tooltip="Proyecto del que se exportan las anotaciones para construir el dataset de entrenamiento. Debe tener tareas con bounding boxes correctamente anotadas."
                        ></field-label>
                        <v-select
                          v-model="trainForm.project"
                          :items="trainProjects"
                          item-title="label"
                          item-value="id"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-select>
                      </v-col>
                      <v-col cols="12">
                        <field-label
                          label="Modelo base"
                          tooltip="Modelo YOLO preentrenado desde el que se inicia el fine-tuning. Modelos más grandes (m, l, x) son más lentos pero potencialmente más precisos. Elegí el mismo tamaño que el modelo en producción para que los pesos sean compatibles."
                        ></field-label>
                        <v-select
                          v-model="trainForm.model_path"
                          :items="trainModels"
                          item-title="label"
                          item-value="path"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-select>
                      </v-col>
                      <v-col cols="12" sm="6">
                        <field-label
                          label="Device"
                          tooltip="Dispositivo de cómputo para el entrenamiento.&#10;&#10;auto: selecciona GPU CUDA si está disponible, sino CPU.&#10;cpu: sin GPU, mucho más lento (puede ser 10-50× más lento que GPU).&#10;0, 1, ...: índice de GPU específica si hay varias instaladas."
                        ></field-label>
                        <v-select
                          v-model="trainForm.device"
                          :items="trainDevices"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-select>
                      </v-col>
                      <v-col cols="6" sm="3">
                        <field-label
                          label="Epochs"
                          tooltip="Número máximo de pasadas completas sobre el dataset. YOLO suele converger en 50-300 epochs según el dataset. El early stopping (patience) puede detenerlo antes si deja de mejorar."
                        ></field-label>
                        <v-text-field
                          v-model.number="trainForm.epochs"
                          type="number"
                          min="1"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-text-field>
                      </v-col>
                      <v-col cols="6" sm="3">
                        <field-label
                          label="Patience"
                          tooltip="Early stopping: epochs consecutivos sin mejora en mAP50-95 antes de detener automáticamente.&#10;&#10;0 = desactivado (entrena hasta el máximo de epochs).&#10;20 es el default. Para datasets pequeños se puede subir a 50."
                        ></field-label>
                        <v-text-field
                          v-model.number="trainForm.patience"
                          type="number"
                          min="0"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-text-field>
                      </v-col>
                      <v-col cols="6" sm="3">
                        <field-label
                          label="Image size"
                          tooltip="Tamaño (en píxeles) al que se redimensionan las imágenes para entrenamiento e inferencia. Debe ser múltiplo de 32.&#10;&#10;640: default, buen balance velocidad/precisión.&#10;1280: más preciso en objetos pequeños, pero usa el doble de VRAM.&#10;Debe coincidir con el tamaño usado en inferencia."
                        ></field-label>
                        <v-text-field
                          v-model.number="trainForm.imgsz"
                          type="number"
                          min="32"
                          step="32"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-text-field>
                      </v-col>
                      <v-col cols="6" sm="3">
                        <field-label
                          label="Batch"
                          tooltip="Imágenes procesadas en paralelo por paso de gradiente. Más grande = gradiente más estable y entrenamiento más rápido, pero más VRAM.&#10;&#10;Si aparece 'CUDA out of memory', reducir a la mitad.&#10;-1: auto (YOLO elige según VRAM disponible)."
                        ></field-label>
                        <v-text-field
                          v-model.number="trainForm.batch"
                          type="number"
                          min="1"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-text-field>
                      </v-col>
                      <v-col cols="6" sm="3">
                        <field-label
                          label="Workers"
                          tooltip="Procesos paralelos del DataLoader para cargar imágenes desde disco. Más workers = carga más rápida si el disco es el cuello de botella.&#10;&#10;Usá 0 si el entrenamiento falla al iniciar — algunos entornos Docker no soportan multiprocessing con fork."
                        ></field-label>
                        <v-text-field
                          v-model.number="trainForm.workers"
                          type="number"
                          min="0"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-text-field>
                      </v-col>
                      <v-col cols="6" sm="3">
                        <field-label
                          label="LR inicial (lr0)"
                          tooltip="Learning rate inicial del optimizador SGD. YOLO aplica warmup lineal en las primeras epochs y luego decaimiento cosine hasta lr_f.&#10;&#10;0.01: default para fine-tuning desde cero.&#10;Demasiado alto: entrenamiento inestable (loss diverge).&#10;Demasiado bajo: convergencia muy lenta."
                        ></field-label>
                        <v-text-field
                          v-model.number="trainForm.lr0"
                          type="number"
                          min="0"
                          step="0.0001"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-text-field>
                      </v-col>
                      <v-col cols="6" sm="3">
                        <field-label
                          label="Weight decay"
                          tooltip="Regularización L2 del optimizador: penaliza pesos grandes para reducir sobreajuste.&#10;&#10;0.0005: default de YOLO, funciona bien en la mayoría de casos.&#10;Aumentarlo (0.001) si el modelo sobreajusta.&#10;Reducirlo (0.00001) si el dataset es muy pequeño."
                        ></field-label>
                        <v-text-field
                          v-model.number="trainForm.weight_decay"
                          type="number"
                          min="0"
                          step="0.00001"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-text-field>
                      </v-col>
                      <v-col cols="6" sm="3">
                        <field-label
                          label="Cosine LR"
                          tooltip="Activa el scheduler cosine annealing: el learning rate decae siguiendo una curva coseno desde lr0 hasta lr_f.&#10;&#10;Recomendado para entrenamientos largos (>100 epochs). Produce convergencia más suave que el decay lineal."
                        ></field-label>
                        <v-select
                          v-model="trainForm.cos_lr"
                          :items="[{{title:'No',value:false}},{{title:'Sí',value:true}}]"
                          item-title="title"
                          item-value="value"
                          variant="outlined"
                          density="compact"
                          hide-details
                        ></v-select>
                      </v-col>
                      <v-col cols="12">
                        <field-label
                          label="Train split"
                          tooltip="Porcentaje de imágenes del proyecto que se usan para entrenamiento. El resto se reserva para validación.&#10;&#10;80/20: default recomendado con datasets medianos (>500 imágenes).&#10;Con datasets muy pequeños (<200) puede subirse a 90/10, aunque la señal de validación será más ruidosa."
                        ></field-label>
                        <div class="text-caption text-medium-emphasis mb-1">
                          <strong>{{{{ trainForm.train_percent }}}}%</strong> train /
                          <strong>{{{{ 100 - trainForm.train_percent }}}}%</strong> valid
                        </div>
                        <v-slider
                          v-model="trainForm.train_percent"
                          min="1" max="99" step="1"
                          color="primary"
                          thumb-label
                          density="compact"
                          hide-details
                        ></v-slider>
                      </v-col>
                    </v-row>

                    <!-- Upload modelo externo -->
                    <v-divider class="my-3"></v-divider>
                    <div class="text-caption text-medium-emphasis mb-2">Modelo externo (.pt)</div>
                    <div class="d-flex align-center ga-2">
                      <v-file-input
                        v-model="externalModelFile"
                        label="Subir .pt como modelo base"
                        accept=".pt"
                        variant="outlined"
                        density="compact"
                        prepend-icon=""
                        hide-details
                        style="flex:1"
                      ></v-file-input>
                      <v-btn
                        variant="outlined"
                        :loading="uploadingModel"
                        :disabled="!externalModelFile || !externalModelFile.length"
                        @click="uploadExternalModel"
                        size="small"
                      >Subir</v-btn>
                    </div>
                    <div v-if="uploadMsg" class="text-caption mt-1" :class="uploadOk ? 'text-success' : 'text-error'">
                      {{{{ uploadMsg }}}}
                    </div>

                    <v-divider class="my-4"></v-divider>
                    <v-btn
                      color="primary"
                      variant="flat"
                      :loading="startingTrain"
                      :disabled="isRunning || !trainForm.project"
                      @click="startTraining"
                      prepend-icon="mdi-play"
                      block
                    >Iniciar entrenamiento</v-btn>
                    <div v-if="startTrainMsg" class="text-caption mt-2" :class="startTrainOk ? 'text-success' : 'text-error'">
                      {{{{ startTrainMsg }}}}
                    </div>
                  </v-card-text>
                </v-card>
              </v-col>

              <!-- Card progreso live (solo si hay job running/queued) -->
              <v-col cols="12" md="6">
                <v-card v-if="runningJob" variant="outlined" color="warning" class="mb-4">
                  <v-card-title class="text-body-1 font-weight-bold pt-4 pb-1">
                    <v-icon class="mr-2 mdi-spin" color="warning">mdi-loading</v-icon>
                    Entrenamiento en curso
                  </v-card-title>
                  <v-card-text>
                    <div class="text-caption text-medium-emphasis mb-1">
                      Proyecto {{{{ runningJob.project }}}}
                    </div>
                    <div class="d-flex align-center justify-space-between mb-1">
                      <span class="text-body-2">
                        Época {{{{ runningJob.progress && runningJob.progress.current_epoch || 0 }}}}
                        / {{{{ runningJob.progress && runningJob.progress.total_epochs || '—' }}}}
                      </span>
                      <span class="text-caption text-medium-emphasis">{{{{ fmtDuration(runningJob.elapsed_seconds) }}}}</span>
                    </div>
                    <v-progress-linear
                      :model-value="epochPct(runningJob)"
                      color="warning"
                      height="8"
                      rounded
                      class="mb-3"
                    ></v-progress-linear>

                    <div class="d-flex ga-2 flex-wrap mb-3">
                      <v-chip color="success" size="small" variant="tonal">
                        Best epoch: {{{{ runningJob.metrics && runningJob.metrics.summary && runningJob.metrics.summary.best_epoch !== null ? runningJob.metrics.summary.best_epoch : '—' }}}}
                      </v-chip>
                      <v-chip color="primary" size="small" variant="tonal">
                        mAP50-95: {{{{ runningJob.metrics && runningJob.metrics.summary && runningJob.metrics.summary.best_metric !== null ? Number(runningJob.metrics.summary.best_metric).toFixed(4) : '—' }}}}
                      </v-chip>
                      <v-chip size="small" variant="tonal">
                        {{{{ runningJob.phase || runningJob.status }}}}
                      </v-chip>
                    </div>

                    <div class="text-caption text-medium-emphasis mb-4">{{{{ runningJob.message }}}}</div>

                    <v-btn
                      color="error"
                      variant="outlined"
                      size="small"
                      @click="cancelJob(runningJob.id)"
                      prepend-icon="mdi-stop"
                    >Cancelar entrenamiento</v-btn>
                  </v-card-text>
                </v-card>

                <v-alert v-else type="info" variant="tonal">
                  Sin entrenamiento activo. Completá el formulario y hacé clic en "Iniciar entrenamiento".
                </v-alert>
              </v-col>

            </v-row>
          </v-tabs-window-item>

          <!-- ═════════════════ TAB: HISTORIAL ═════════════════ -->
          <v-tabs-window-item value="historial">
            <v-alert v-if="jobs.length === 0" type="info" variant="tonal" class="mt-2">
              Todavía no hay jobs de entrenamiento.
            </v-alert>
            <v-row v-else>

              <!-- ── Lista de jobs (izquierda) ── -->
              <v-col cols="12" md="4" lg="3">
                <v-list density="compact" nav>
                  <v-list-item
                    v-for="job in jobs" :key="job.id"
                    :active="selectedJobId === job.id"
                    @click="selectJob(job)"
                    rounded="lg"
                    class="mb-1 pa-2"
                  >
                    <template v-slot:prepend>
                      <v-icon :color="statusColor(job.status)" size="18" class="mr-2">
                        {{{{ statusIcon(job.status) }}}}
                      </v-icon>
                    </template>
                    <v-list-item-title class="text-body-2 font-weight-medium">
                      Proyecto {{{{ job.project }}}}
                    </v-list-item-title>
                    <v-list-item-subtitle class="text-caption">
                      <v-chip :color="statusColor(job.status)" size="x-small" variant="tonal" class="mr-1">
                        {{{{ job.status }}}}
                      </v-chip>
                      {{{{ fmtDuration(job.elapsed_seconds) }}}}
                    </v-list-item-subtitle>
                    <v-list-item-subtitle v-if="job.metrics && job.metrics.summary && job.metrics.summary.best_metric != null" class="text-caption mt-1">
                      Best ep.{{{{ job.metrics.summary.best_epoch }}}} · mAP {{{{ Number(job.metrics.summary.best_metric).toFixed(4) }}}}
                    </v-list-item-subtitle>
                    <template v-slot:append>
                      <v-tooltip text="Eliminar job" location="top">
                        <template v-slot:activator="{{ props }}">
                          <v-btn
                            v-bind="props"
                            icon size="x-small" variant="text" color="error"
                            :disabled="job.status === 'running'"
                            @click.stop="deleteJob(job.id)"
                          >
                            <v-icon size="16">mdi-delete-outline</v-icon>
                          </v-btn>
                        </template>
                      </v-tooltip>
                    </template>
                  </v-list-item>
                </v-list>
              </v-col>

              <!-- ── Detalle del job (derecha) ── -->
              <v-col cols="12" md="8" lg="9">
                <v-alert v-if="!selectedJob" type="info" variant="tonal">
                  Seleccioná un job para ver el detalle.
                </v-alert>

                <div v-else>

                  <!-- Resumen -->
                  <v-card variant="outlined" class="mb-3">
                    <v-card-title class="text-body-2 font-weight-bold pt-3 pb-1 d-flex align-center justify-space-between">
                      <span>
                        <v-icon size="16" class="mr-1" color="primary">mdi-information-outline</v-icon>
                        Resumen · job {{{{ selectedJob.id ? selectedJob.id.substring(0,8) : '' }}}}
                      </span>
                      <div class="d-flex ga-1">
                        <v-btn
                          v-if="selectedJob.trained_model"
                          size="x-small" variant="tonal" color="success"
                          prepend-icon="mdi-download"
                          :href="'/download/models/' + selectedJob.trained_model.split('/').pop()"
                        >Descargar</v-btn>
                        <v-btn
                          v-if="selectedJob.trained_model"
                          size="x-small" variant="tonal" color="primary"
                          prepend-icon="mdi-play"
                          @click="useForInference(selectedJob.trained_model)"
                        >Usar para inferencia</v-btn>
                      </div>
                    </v-card-title>
                    <v-card-text class="pa-0">
                      <table style="width:100%;border-collapse:collapse" class="nv-table">
                        <tbody>
                          <tr>
                            <td class="text-caption text-medium-emphasis" style="width:140px;padding:6px 16px">Status</td>
                            <td style="padding:6px 16px">
                              <v-chip :color="statusColor(selectedJob.status)" size="x-small" variant="tonal">
                                {{{{ selectedJob.status }}}}
                              </v-chip>
                            </td>
                          </tr>
                          <tr>
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Proyecto LS</td>
                            <td class="text-caption" style="padding:6px 16px">{{{{ selectedJob.project }}}}</td>
                          </tr>
                          <tr v-if="selectedJob.train_config && selectedJob.train_config.model_path">
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Modelo base</td>
                            <td class="text-caption" style="padding:6px 16px">{{{{ selectedJob.train_config.model_path.split('/').pop() }}}}</td>
                          </tr>
                          <tr v-if="selectedJob.train_config && selectedJob.train_config.device">
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Device</td>
                            <td class="text-caption" style="padding:6px 16px">{{{{ selectedJob.train_config.device }}}}</td>
                          </tr>
                          <tr v-if="selectedJob.dataset">
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Dataset</td>
                            <td class="text-caption" style="padding:6px 16px">
                              {{{{ selectedJob.dataset.train_images || '?' }}}} train /
                              {{{{ selectedJob.dataset.val_images || '?' }}}} val
                              <span v-if="selectedJob.dataset.train_percent"> ({{{{ selectedJob.dataset.train_percent }}}}% train)</span>
                            </td>
                          </tr>
                          <tr v-if="selectedJob.started_at">
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Iniciado</td>
                            <td class="text-caption" style="padding:6px 16px">{{{{ fmtDate(selectedJob.started_at) }}}}</td>
                          </tr>
                          <tr v-if="selectedJob.finished_at">
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Finalizado</td>
                            <td class="text-caption" style="padding:6px 16px">{{{{ fmtDate(selectedJob.finished_at) }}}}</td>
                          </tr>
                          <tr>
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Duración</td>
                            <td class="text-caption" style="padding:6px 16px">{{{{ fmtDuration(selectedJob.elapsed_seconds) }}}}</td>
                          </tr>
                          <tr v-if="selectedJob.trained_model">
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Modelo resultado</td>
                            <td class="text-caption" style="padding:6px 16px">{{{{ selectedJob.trained_model.split('/').pop() }}}}</td>
                          </tr>
                          <tr v-if="selectedJob.message">
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Mensaje</td>
                            <td class="text-caption" style="padding:6px 16px">{{{{ selectedJob.message }}}}</td>
                          </tr>
                          <tr v-if="selectedJob.error">
                            <td class="text-caption text-medium-emphasis" style="padding:6px 16px">Error</td>
                            <td class="text-caption text-error" style="padding:6px 16px">{{{{ selectedJob.error }}}}</td>
                          </tr>
                        </tbody>
                      </table>
                    </v-card-text>
                  </v-card>

                  <!-- Best y paciencia -->
                  <v-card v-if="selectedJob.metrics && selectedJob.metrics.summary && Object.keys(selectedJob.metrics.summary).length" variant="outlined" class="mb-3">
                    <v-card-title class="text-body-2 font-weight-bold pt-3 pb-2">
                      <v-icon size="16" class="mr-1" color="success">mdi-trophy-outline</v-icon>
                      Best y paciencia
                    </v-card-title>
                    <v-card-text>
                      <div class="d-flex ga-2 flex-wrap mb-3">
                        <v-chip color="success" size="small" variant="tonal" prepend-icon="mdi-star">
                          Best epoch: {{{{ selectedJob.metrics.summary.best_epoch != null ? selectedJob.metrics.summary.best_epoch : '—' }}}}
                        </v-chip>
                        <v-chip color="success" size="small" variant="tonal" prepend-icon="mdi-chart-line">
                          Best mAP50-95: {{{{ selectedJob.metrics.summary.best_metric != null ? Number(selectedJob.metrics.summary.best_metric).toFixed(4) : '—' }}}}
                        </v-chip>
                        <v-chip size="small" variant="tonal">
                          Epoch actual: {{{{ selectedJob.metrics.summary.current_epoch != null ? selectedJob.metrics.summary.current_epoch : '—' }}}}
                        </v-chip>
                        <v-chip :color="patienceColor(selectedJob)" size="small" variant="tonal">
                          Sin mejora: {{{{ selectedJob.metrics.summary.epochs_without_improvement != null ? selectedJob.metrics.summary.epochs_without_improvement : '—' }}}} / {{{{ selectedJob.metrics.summary.patience }}}}
                        </v-chip>
                      </div>
                      <div class="text-caption text-medium-emphasis mb-1">Paciencia consumida</div>
                      <v-progress-linear
                        :model-value="patiencePct(selectedJob)"
                        :color="patienceColor(selectedJob)"
                        height="8"
                        rounded
                      ></v-progress-linear>
                    </v-card-text>
                  </v-card>

                  <!-- Métricas YOLO (14 valores) -->
                  <v-card v-if="selectedJob.metrics && selectedJob.metrics.latest && Object.keys(selectedJob.metrics.latest).length" variant="outlined" class="mb-3">
                    <v-card-title class="text-body-2 font-weight-bold pt-3 pb-2">
                      <v-icon size="16" class="mr-1" color="warning">mdi-gauge</v-icon>
                      Métricas YOLO — última época
                    </v-card-title>
                    <v-card-text>
                      <div class="d-flex ga-2 flex-wrap">
                        <v-chip
                          v-for="[key, label] in metricLabels" :key="key"
                          :color="key === 'metrics/mAP50-95(B)' ? 'success' : undefined"
                          :size="key === 'metrics/mAP50-95(B)' ? 'default' : 'small'"
                          :variant="key === 'metrics/mAP50-95(B)' ? 'flat' : 'tonal'"
                        >
                          {{{{ label }}}}: {{{{ selectedJob.metrics.latest[key] || '—' }}}}
                        </v-chip>
                      </div>
                    </v-card-text>
                  </v-card>

                  <!-- Curvas SVG -->
                  <v-card v-if="selectedJob.metrics && selectedJob.metrics.rows && selectedJob.metrics.rows.length > 1" variant="outlined" class="mb-3">
                    <v-card-title class="text-body-2 font-weight-bold pt-3 pb-2">
                      <v-icon size="16" class="mr-1" color="info">mdi-chart-bell-curve-cumulative</v-icon>
                      Curvas de entrenamiento
                    </v-card-title>
                    <v-card-text>
                      <div class="chart-grid" v-html="chartsHtml"></div>
                    </v-card-text>
                  </v-card>

                </div>
              </v-col>
            </v-row>

            <!-- Tabla de modelos entrenados -->
            <v-card v-if="modelsTable.length" variant="outlined" class="mt-4">
              <v-card-title class="text-body-2 font-weight-bold pt-3 pb-1">
                <v-icon size="16" class="mr-1" color="primary">mdi-file-cog-outline</v-icon>
                Modelos entrenados ({{{{ modelsTable.length }}}})
              </v-card-title>
              <v-card-text class="pa-0">
                <table style="width:100%;border-collapse:collapse" class="nv-table">
                  <thead>
                    <tr>
                      <th class="text-caption">Nombre</th>
                      <th class="text-caption">Tamaño</th>
                      <th class="text-caption">mAP50-95</th>
                      <th class="text-caption">Proyecto</th>
                      <th class="text-caption">Fecha</th>
                      <th class="text-caption">Acciones</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="m in modelsTable" :key="m.path">
                      <td class="text-caption">
                        <v-chip v-if="m.active" color="success" size="x-small" variant="flat" class="mr-1">activo</v-chip>
                        {{{{ m.name }}}}
                      </td>
                      <td class="text-caption">{{{{ m.sizeLabel }}}}</td>
                      <td class="text-caption">{{{{ m.map5095 }}}}</td>
                      <td class="text-caption">{{{{ m.project }}}}</td>
                      <td class="text-caption">{{{{ m.modifiedLabel }}}}</td>
                      <td>
                        <v-tooltip text="Descargar modelo (.pt)" location="top">
                          <template v-slot:activator="{{ props }}">
                            <v-btn
                              v-bind="props"
                              icon size="x-small" variant="text"
                              :href="'/download/models/' + m.name" target="_blank"
                            >
                              <v-icon size="16">mdi-download</v-icon>
                            </v-btn>
                          </template>
                        </v-tooltip>
                        <v-tooltip text="Activar para inferencia y cambiar al tab Inferencia" location="top">
                          <template v-slot:activator="{{ props }}">
                            <v-btn
                              v-bind="props"
                              icon size="x-small" variant="text" color="primary"
                              @click="useForInference(m.path)"
                            >
                              <v-icon size="16">mdi-play</v-icon>
                            </v-btn>
                          </template>
                        </v-tooltip>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </v-card-text>
            </v-card>

          </v-tabs-window-item>

        </v-tabs-window>
      </v-container>
    </v-main>

  </v-app>
</div>

<script>const INITIAL_DATA = {initial_data};</script>
<script src="https://cdn.jsdelivr.net/npm/vue@3/dist/vue.global.prod.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vuetify@3/dist/vuetify.min.js"></script>
<script>
const {{ createApp, ref, computed, watch, onMounted, onUnmounted }} = Vue;
const {{ createVuetify }} = Vuetify;

// Componente reutilizable: label + tooltip de ayuda (igual que lpr-ocr-labeler)
const FieldLabel = {{
  props: {{ label: String, tooltip: String }},
  template: `
    <div class="d-flex align-center mb-1">
      <span style="font-size:12px;font-weight:600;color:rgba(255,255,255,.6)">{{{{ label }}}}</span>
      <v-tooltip v-if="tooltip" :text="tooltip" location="top end" max-width="320" open-delay="150">
        <template v-slot:activator="{{ props }}">
          <v-icon v-bind="props" size="14" color="grey-darken-1" style="margin-left:4px;cursor:help">
            mdi-information-outline
          </v-icon>
        </template>
      </v-tooltip>
    </div>
  `,
}};

const vuetify = createVuetify({{
  theme: {{
    defaultTheme: 'dark',
    themes: {{
      dark: {{
        dark: true,
        colors: {{
          background: '#121212',
          surface: '#1e1e1e',
          primary: '#1976D2',
          success: '#4CAF50',
          warning: '#FF9800',
          error: '#ef5350',
        }},
      }},
    }},
  }},
  icons: {{ defaultSet: 'mdi' }},
}});

createApp({{
  setup() {{
    const tab = ref('historial');
    const jobs = ref([]);
    const selectedJobId = ref(null);
    const selectedJob = ref(null);
    const activeModelName = ref(INITIAL_DATA.currentModelName || '');
    const appVersion = ref(INITIAL_DATA.appVersion || '');

    // ── Entrenar ──
    const TRAIN_STORAGE_KEY = 'yolo11.trainForm';
    function loadSavedForm() {{
      try {{ return JSON.parse(localStorage.getItem(TRAIN_STORAGE_KEY) || 'null'); }}
      catch {{ return null; }}
    }}
    function saveForm(form) {{
      try {{ localStorage.setItem(TRAIN_STORAGE_KEY, JSON.stringify(form)); }} catch {{}}
    }}
    const _saved = loadSavedForm();
    const trainForm = ref({{
      project: _saved && _saved.project != null ? _saved.project : null,
      model_path: (_saved && _saved.model_path) || '',
      device: (_saved && _saved.device) || INITIAL_DATA.defaults.device || 'auto',
      epochs: (_saved && _saved.epochs != null) ? _saved.epochs : INITIAL_DATA.defaults.epochs,
      imgsz: (_saved && _saved.imgsz != null) ? _saved.imgsz : INITIAL_DATA.defaults.imgsz,
      batch: (_saved && _saved.batch != null) ? _saved.batch : INITIAL_DATA.defaults.batch,
      patience: (_saved && _saved.patience != null) ? _saved.patience : INITIAL_DATA.defaults.patience,
      workers: (_saved && _saved.workers != null) ? _saved.workers : INITIAL_DATA.defaults.workers,
      lr0: (_saved && _saved.lr0 != null) ? _saved.lr0 : INITIAL_DATA.defaults.lr0,
      weight_decay: (_saved && _saved.weight_decay != null) ? _saved.weight_decay : INITIAL_DATA.defaults.weightDecay,
      cos_lr: (_saved && _saved.cos_lr != null) ? _saved.cos_lr : INITIAL_DATA.defaults.cosLr,
      train_percent: (_saved && _saved.train_percent != null) ? _saved.train_percent : INITIAL_DATA.defaults.splitPercent,
    }});
    const trainProjects = ref(
      (INITIAL_DATA.projects || []).map(p => ({{
        id: p.id,
        label: p.id + ' — ' + p.title + (p.task_count ? ' (' + p.task_count + ' imágenes)' : ''),
      }}))
    );
    watch(trainForm, (val) => saveForm(val), {{ deep: true }});

    const trainDevices = ref(
      ['auto', 'cpu'].concat((INITIAL_DATA.devices || []).map((name, i) => ({{ title: i + ' — ' + name, value: String(i) }})))
    );
    const trainModels = ref([]);
    const modelsTable = ref([]);
    const externalModelFile = ref(null);
    const uploadingModel = ref(false);
    const uploadMsg = ref('');
    const uploadOk = ref(true);
    const startingTrain = ref(false);
    const startTrainMsg = ref('');
    const startTrainOk = ref(true);
    const runningJob = computed(() => jobs.value.find(j => j.status === 'running' || j.status === 'queued') || null);

    // ── Inferencia ──
    const availableModels = ref([]);
    const inferenceModelPath = ref(INITIAL_DATA.currentModelPath || '');
    const activeLabels = ref([]);
    const confidence = ref(INITIAL_DATA.confidenceThreshold || 0.25);
    const ultralytics = ref(INITIAL_DATA.ultralytics || '');
    const applyingModel = ref(false);
    const inferenceMsg = ref('');
    const inferenceMsgType = ref('success');

    const isRunning = computed(() =>
      jobs.value.some(j => j.status === 'running' || j.status === 'queued')
    );

    function statusColor(status) {{
      return {{ running: 'warning', queued: 'info', completed: 'success', failed: 'error', cancelled: 'grey' }}[status] || 'grey';
    }}
    function statusIcon(status) {{
      return {{ running: 'mdi-loading', queued: 'mdi-clock-outline', completed: 'mdi-check-circle', failed: 'mdi-alert-circle', cancelled: 'mdi-cancel' }}[status] || 'mdi-help-circle';
    }}
    function fmtDuration(secs) {{
      secs = Math.max(Math.floor(secs || 0), 0);
      const h = Math.floor(secs / 3600); secs %= 3600;
      const m = Math.floor(secs / 60); const s = secs % 60;
      return (h ? h + 'h ' : '') + (h || m ? m + 'm ' : '') + s + 's';
    }}

    function epochPct(job) {{
      if (!job || !job.progress) return 0;
      const {{ current_epoch, total_epochs }} = job.progress;
      if (!total_epochs) return 0;
      return Math.round((current_epoch / total_epochs) * 100);
    }}

    async function uploadExternalModel() {{
      if (!externalModelFile.value || !externalModelFile.value.length) return;
      uploadingModel.value = true; uploadMsg.value = '';
      try {{
        const fd = new FormData();
        fd.append('model', externalModelFile.value[0]);
        const res = await fetch('/api/external-models', {{ method: 'POST', body: fd }});
        const data = await res.json().catch(() => ({{}}));
        if (res.ok) {{
          uploadMsg.value = 'Modelo subido: ' + data.model.name;
          uploadOk.value = true;
          trainForm.value.model_path = data.model.path;
          await fetchModels();
        }} else {{
          uploadMsg.value = 'Error: ' + (data.message || JSON.stringify(data));
          uploadOk.value = false;
        }}
      }} catch (e) {{
        uploadMsg.value = 'Error de red: ' + e.message;
        uploadOk.value = false;
      }} finally {{
        uploadingModel.value = false;
      }}
    }}

    async function startTraining() {{
      startingTrain.value = true; startTrainMsg.value = '';
      try {{
        const payload = {{
          project: Number(trainForm.value.project),
          model_path: trainForm.value.model_path,
          device: trainForm.value.device,
          epochs: Number(trainForm.value.epochs),
          imgsz: Number(trainForm.value.imgsz),
          batch: Number(trainForm.value.batch),
          patience: Number(trainForm.value.patience),
          workers: Number(trainForm.value.workers),
          lr0: Number(trainForm.value.lr0),
          weight_decay: Number(trainForm.value.weight_decay),
          cos_lr: Boolean(trainForm.value.cos_lr),
          train_percent: Number(trainForm.value.train_percent),
        }};
        const res = await fetch('/train', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload),
        }});
        const data = await res.json();
        if (res.ok && data.status !== 'busy') {{
          startTrainMsg.value = 'Job iniciado: ' + data.job_id;
          startTrainOk.value = true;
          await fetchJobs();
        }} else if (data.status === 'busy') {{
          startTrainMsg.value = 'Ya hay un entrenamiento corriendo.';
          startTrainOk.value = false;
        }} else {{
          startTrainMsg.value = 'Error: ' + (data.message || JSON.stringify(data));
          startTrainOk.value = false;
        }}
      }} catch (e) {{
        startTrainMsg.value = 'Error de red: ' + e.message;
        startTrainOk.value = false;
      }} finally {{
        startingTrain.value = false;
      }}
    }}

    async function cancelJob(jobId) {{
      const res = await fetch('/api/jobs/' + jobId + '/cancel', {{ method: 'POST' }});
      const data = await res.json().catch(() => ({{}}));
      if (res.ok) {{
        startTrainMsg.value = 'Entrenamiento cancelado. El backend se reiniciará.';
        startTrainOk.value = true;
        setTimeout(fetchJobs, 2500);
      }} else {{
        startTrainMsg.value = 'Error al cancelar: ' + (data.message || '');
        startTrainOk.value = false;
      }}
    }}

    async function fetchModels() {{
      try {{
        const res = await fetch('/api/models');
        const data = await res.json();
        const opts = data.available_models || [];
        availableModels.value = opts.map(m => ({{
          ...m,
          label: m.name + (m.source === 'entrenado' && m.project ? ' [proy ' + m.project + ']' : '') + (m.active ? ' ✓' : ''),
        }}));
        trainModels.value = opts.map(m => ({{
          path: m.path,
          label: m.name + (m.source === 'entrenado' && m.project ? ' [proy ' + m.project + ']' : ''),
        }}));
        const active = opts.find(m => m.active);
        if (active) {{
          activeModelName.value = active.name;
          inferenceModelPath.value = active.path;
          if (!trainForm.value.model_path) trainForm.value.model_path = active.path;
        }}
        modelsTable.value = (data.models || []).map(m => {{
          const lat = (m.metadata && m.metadata.metrics && m.metadata.metrics.latest) || {{}};
          return {{
            ...m,
            map5095: lat['metrics/mAP50-95(B)'] != null ? Number(lat['metrics/mAP50-95(B)']).toFixed(4) : '—',
            project: (m.metadata && m.metadata.dataset && m.metadata.dataset.project) || '—',
            modifiedLabel: m.modified_at ? new Date(m.modified_at * 1000).toLocaleDateString('es-AR') : '—',
            sizeLabel: fmtSize(m.size),
          }};
        }});
      }} catch (e) {{
        console.warn('Error fetching models', e);
      }}
    }}

    async function fetchStatus() {{
      try {{
        const res = await fetch('/status');
        const data = await res.json();
        if (Array.isArray(data.labels)) activeLabels.value = data.labels;
      }} catch (e) {{
        console.warn('Error fetching status', e);
      }}
    }}

    async function applyInferenceModel() {{
      applyingModel.value = true;
      inferenceMsg.value = '';
      try {{
        const res = await fetch('/api/active-model', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ model_path: inferenceModelPath.value }}),
        }});
        const data = await res.json();
        if (res.ok) {{
          inferenceMsg.value = 'Modelo activado: ' + (data.active_model_path || inferenceModelPath.value);
          inferenceMsgType.value = 'success';
          activeModelName.value = inferenceModelPath.value.split('/').pop();
          await fetchModels();
        }} else {{
          inferenceMsg.value = 'Error: ' + (data.message || JSON.stringify(data));
          inferenceMsgType.value = 'error';
        }}
      }} catch (e) {{
        inferenceMsg.value = 'Error de red: ' + e.message;
        inferenceMsgType.value = 'error';
      }} finally {{
        applyingModel.value = false;
      }}
    }}

    async function fetchJobs() {{
      try {{
        const res = await fetch('/api/jobs');
        const data = await res.json();
        jobs.value = data.jobs || [];
        if (selectedJobId.value) {{
          const found = jobs.value.find(j => j.id === selectedJobId.value);
          if (found) selectedJob.value = found;
        }}
      }} catch (e) {{
        console.warn('Error fetching jobs', e);
      }}
    }}

    async function selectJob(job) {{
      selectedJobId.value = job.id;
      try {{
        const res = await fetch('/api/jobs/' + job.id);
        selectedJob.value = await res.json();
      }} catch (e) {{
        selectedJob.value = job;
      }}
    }}

    function fmtSize(bytes) {{
      if (!bytes) return '—';
      if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
      if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return bytes + ' B';
    }}

    function fmtDate(ts) {{
      if (!ts) return '—';
      try {{
        const d = new Date(typeof ts === 'number' ? ts * 1000 : ts);
        return d.toLocaleString('es-AR', {{ day:'2-digit', month:'2-digit', year:'numeric', hour:'2-digit', minute:'2-digit' }});
      }} catch(e) {{ return String(ts); }}
    }}

    async function deleteJob(id) {{
      const res = await fetch('/api/jobs/' + id, {{ method: 'DELETE' }});
      if (res.ok) {{
        if (selectedJobId.value === id) {{ selectedJobId.value = null; selectedJob.value = null; }}
        await fetchJobs();
      }}
    }}

    function useForInference(path) {{
      inferenceModelPath.value = path;
      tab.value = 'inferencia';
      applyInferenceModel();
    }}

    function patiencePct(job) {{
      const s = ((job && job.metrics) ? job.metrics.summary : null) || {{}};
      if (!s.patience) return 0;
      return Math.round(((s.epochs_without_improvement || 0) / s.patience) * 100);
    }}

    function patienceColor(job) {{
      const pct = patiencePct(job);
      if (pct >= 80) return 'error';
      if (pct >= 50) return 'warning';
      return 'success';
    }}

    const metricLabels = [
      ['metrics/mAP50-95(B)', 'mAP50-95'],
      ['metrics/mAP50(B)', 'mAP50'],
      ['metrics/precision(B)', 'Precisión'],
      ['metrics/recall(B)', 'Recall'],
      ['train/box_loss', 'Train box loss'],
      ['train/cls_loss', 'Train cls loss'],
      ['train/dfl_loss', 'Train dfl loss'],
      ['val/box_loss', 'Val box loss'],
      ['val/cls_loss', 'Val cls loss'],
      ['val/dfl_loss', 'Val dfl loss'],
    ];

    function svgChart(rows, key, title, color) {{
      const W = 560, H = 180, PAD = 26;
      const values = rows.map(r => {{
        const v = parseFloat(r[key]);
        return isNaN(v) ? null : v;
      }}).filter(v => v !== null);
      if (values.length < 2) return '<div style="opacity:0.4;text-align:center;padding:24px 0">'
        + '<span style="font-size:12px;color:rgba(255,255,255,.4)">Sin datos — ' + title + '</span></div>';
      const minV = Math.min(...values), maxV = Math.max(...values);
      const span = Math.max(maxV - minV, 1e-9);
      const pts = values.map((v, i) => {{
        const x = PAD + i * ((W - PAD * 2) / (values.length - 1));
        const y = H - PAD - ((v - minV) / span) * (H - PAD * 2);
        return x.toFixed(1) + ',' + y.toFixed(1);
      }}).join(' ');
      const latest = values[values.length - 1];
      return '<div class="chart-wrap">'
        + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
        + '<span style="font-size:11px;font-weight:600;color:rgba(255,255,255,.7)">' + title + '</span>'
        + '<span style="font-size:11px;color:rgba(255,255,255,.5)">último: ' + latest.toFixed(4) + '</span>'
        + '</div>'
        + '<svg viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;height:130px;display:block">'
        + '<line x1="' + PAD + '" y1="' + (H-PAD) + '" x2="' + (W-PAD) + '" y2="' + (H-PAD) + '" stroke="rgba(255,255,255,.1)" stroke-width="1"/>'
        + '<line x1="' + PAD + '" y1="' + PAD + '" x2="' + PAD + '" y2="' + (H-PAD) + '" stroke="rgba(255,255,255,.1)" stroke-width="1"/>'
        + '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        + '</svg>'
        + '<div style="display:flex;justify-content:space-between">'
        + '<span style="font-size:10px;color:rgba(255,255,255,.3)">min ' + minV.toFixed(4) + '</span>'
        + '<span style="font-size:10px;color:rgba(255,255,255,.3)">max ' + maxV.toFixed(4) + '</span>'
        + '</div></div>';
    }}

    const chartsHtml = computed(() => {{
      const job = selectedJob.value;
      if (!job || !job.metrics || !job.metrics.rows || job.metrics.rows.length < 2) return '';
      const rows = job.metrics.rows;
      const defs = [
        {{ key: 'metrics/mAP50-95(B)', title: 'mAP50-95', color: '#4CAF50' }},
        {{ key: 'metrics/mAP50(B)', title: 'mAP50', color: '#1976D2' }},
        {{ key: 'train/box_loss', title: 'Train box loss', color: '#FF9800' }},
        {{ key: 'val/box_loss', title: 'Val box loss', color: '#ef5350' }},
      ];
      return defs.map(c => svgChart(rows, c.key, c.title, c.color)).join('');
    }});

    let pollTimer = null;
    onMounted(async () => {{
      await Promise.all([fetchJobs(), fetchModels(), fetchStatus()]);
      if (jobs.value.length > 0) selectJob(jobs.value[0]);
      pollTimer = setInterval(async () => {{
        await fetchJobs();
        if (selectedJobId.value) {{
          try {{
            const res = await fetch('/api/jobs/' + selectedJobId.value);
            if (res.ok) selectedJob.value = await res.json();
          }} catch(e) {{}}
        }}
        if (isRunning.value) await fetchModels();
      }}, 2000);
    }});
    onUnmounted(() => clearInterval(pollTimer));

    return {{
      tab, jobs, selectedJobId, selectedJob, activeModelName, appVersion,
      isRunning, statusColor, statusIcon, fmtDuration, fmtDate, selectJob,
      availableModels, inferenceModelPath, activeLabels,
      confidence, ultralytics, applyingModel, inferenceMsg, inferenceMsgType,
      applyInferenceModel,
      trainForm, trainProjects, trainDevices, trainModels,
      externalModelFile, uploadingModel, uploadMsg, uploadOk,
      startingTrain, startTrainMsg, startTrainOk, runningJob,
      epochPct, uploadExternalModel, startTraining, cancelJob,
      deleteJob, useForInference, patiencePct, patienceColor,
      metricLabels, chartsHtml,
      modelsTable, fmtSize,
    }};
  }},
}}).use(vuetify).component('field-label', FieldLabel).mount('#app');
</script>
</body>
</html>"""
