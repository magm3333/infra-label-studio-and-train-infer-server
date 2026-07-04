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
            projects.append({"id": project_id, "title": str(title)})
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
        "ultralytics": ULTRALYTICS_VERSION,
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
  <link href="https://cdn.jsdelivr.net/npm/vuetify@3/dist/vuetify.min.css" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/@mdi/font@7/css/materialdesignicons.min.css" rel="stylesheet">
  <style>
    html, body {{ height: 100%; margin: 0; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; }}
    .chart-wrap {{ background: #070b13; border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 12px; }}
    .chart-wrap svg {{ width: 100%; height: 180px; display: block; }}
    .axis {{ stroke: #2b3958; stroke-width: 1; }}
    .v-application {{ background: #121212 !important; }}
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
            <v-alert type="info" variant="tonal">
              Tab Inferencia — próxima fase
            </v-alert>
          </v-tabs-window-item>

          <!-- ═════════════════ TAB: ENTRENAR ═════════════════ -->
          <v-tabs-window-item value="entrenar">
            <v-alert type="info" variant="tonal">
              Tab Entrenar — próxima fase
            </v-alert>
          </v-tabs-window-item>

          <!-- ═════════════════ TAB: HISTORIAL ═════════════════ -->
          <v-tabs-window-item value="historial">
            <v-row v-if="jobs.length === 0">
              <v-col>
                <v-alert type="info" variant="tonal" class="mt-2">
                  Todavía no hay jobs de entrenamiento.
                </v-alert>
              </v-col>
            </v-row>
            <v-row v-else>
              <v-col cols="12" md="4">
                <v-list density="compact" nav>
                  <v-list-item
                    v-for="job in jobs" :key="job.id"
                    :active="selectedJobId === job.id"
                    @click="selectJob(job)"
                    rounded="lg"
                    class="mb-1"
                  >
                    <template v-slot:prepend>
                      <v-icon :color="statusColor(job.status)" size="18">
                        {{{{ statusIcon(job.status) }}}}
                      </v-icon>
                    </template>
                    <v-list-item-title class="text-body-2 font-weight-medium">
                      Proyecto {{{{ job.project }}}}
                    </v-list-item-title>
                    <v-list-item-subtitle class="text-caption">
                      {{{{ job.status }}}} · {{{{ fmtDuration(job.elapsed_seconds) }}}}
                    </v-list-item-subtitle>
                  </v-list-item>
                </v-list>
              </v-col>
              <v-col cols="12" md="8">
                <v-card v-if="selectedJob" variant="outlined">
                  <v-card-title class="text-body-1">
                    Job {{{{ selectedJob.id ? selectedJob.id.substring(0,8) : '' }}}}
                  </v-card-title>
                  <v-card-subtitle>{{{{ selectedJob.status }}}}</v-card-subtitle>
                  <v-card-text>
                    <div class="text-caption text-medium-emphasis">
                      Historial completo — próxima fase
                    </div>
                  </v-card-text>
                </v-card>
                <v-alert v-else type="info" variant="tonal">
                  Seleccioná un job para ver el detalle.
                </v-alert>
              </v-col>
            </v-row>
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
const {{ createApp, ref, computed, onMounted, onUnmounted }} = Vue;
const {{ createVuetify }} = Vuetify;

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

    let pollTimer = null;
    onMounted(async () => {{
      await fetchJobs();
      if (jobs.value.length > 0) selectJob(jobs.value[0]);
      pollTimer = setInterval(fetchJobs, 2000);
    }});
    onUnmounted(() => clearInterval(pollTimer));

    return {{
      tab, jobs, selectedJobId, selectedJob, activeModelName,
      isRunning, statusColor, statusIcon, fmtDuration, selectJob,
    }};
  }},
}}).use(vuetify).mount('#app');
</script>
</body>
</html>"""
