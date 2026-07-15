# model_hub

Driver único de acceso al [Model Hub](https://github.com/OneclickEB/models-hub)
(catálogo público y cifrado de modelos YOLO/LPR OCR entrenados). Resuelve
listar, seleccionar, descargar (con desencriptado transparente) y publicar.

## Instalación

Este paquete se **copia** al proyecto consumidor (no se instala vía pip desde
un índice) — mismo patrón que `ocrkit` en este mismo repo. En un Dockerfile:

```dockerfile
COPY componentes/tools/model-hub-driver/model_hub /app/model_hub
RUN pip install cryptography requests   # o agregar al requirements.txt del proyecto
```

## Configuración (variables de entorno)

| Variable | Requerida para | Descripción |
|---|---|---|
| `MODEL_HUB_REPO` | — | Default `OneclickEB/models-hub`. |
| `MODEL_HUB_CATALOG_URL` | — | Default: raw del `catalog.json` del repo de arriba. |
| `MODEL_HUB_KEY` | descargar (desencriptar), publicar (cifrar) | Passphrase. **Nunca** se commitea ni se loguea. |
| `MODEL_HUB_GH_TOKEN` | publicar | Personal Access Token de GitHub, scope `repo`. |

## Uso — consumidor (listar/descargar)

```python
from model_hub import ModelHubClient

client = ModelHubClient()  # lee env vars

models = client.list_models(family="yolo")
for m in models:
    print(m["model_id"], m["display_name"], len(m["versions"]), "versiones")

# Descarga la última versión, desencripta, verifica integridad, devuelve la ruta lista para usar
path = client.download("yolo-precintos-s", version="latest", format="pt", dest_dir="/models")
```

`client.download()` **no necesita `MODEL_HUB_GH_TOKEN`** — listar el catálogo
y descargar assets de Release son operaciones públicas de solo lectura. Un
consumidor puro (que solo importa modelos, no publica) solo necesita
`MODEL_HUB_KEY_FILE` configurada.

## Patrón recomendado: agregar "Importar del Hub" a un backend que ya sube modelos a mano

Implementado y probado en `ai-ocr-2026/componentes/lpr-admin` (botón "Subir
desde HUB" en la vista de servidores OCR) — usalo como referencia concreta
si vas a repetir el patrón en otro proyecto (ej. neuralVISION, ver más abajo).

1. **Copiar el paquete** a `<tu_backend>/model_hub/` (mismo patrón de
   siempre — se copia al build, no se instala por pip). Agregar
   `cryptography` y `requests` al `requirements.txt` del proyecto (son las
   únicas dependencias externas del driver).
2. **Config**: agregar `MODEL_HUB_KEY_FILE=/run/secrets/model_hub_key.txt`
   al compose del servicio, y montar como volumen la MISMA clave que ya
   usan los demás consumidores/publicadores (una sola passphrase para todo
   el hub — no crear una nueva por proyecto):
   ```yaml
   environment:
     MODEL_HUB_KEY_FILE: /run/secrets/model_hub_key.txt
   volumes:
     - /m2data/proyectos/ai-ocr-2026/componentes/tools/ocr-labeling/secrets/model_hub_key.txt:/run/secrets/model_hub_key.txt:ro
   ```
3. **Endpoint de listado** (proxy fino de `list_models()`, para que el
   frontend no necesite hablar con GitHub directo — evita CORS/auth del
   lado del cliente):
   ```python
   @app.get("/api/model-hub/models")
   def list_hub_models():
       family = request.args.get("family", "lprocr")  # o lo que corresponda a tu dominio
       from model_hub import ModelHubClient
       return jsonify({"models": ModelHubClient().list_models(family=family)})
   ```
4. **Endpoint de importación**: descarga+desencripta a un archivo temporal y
   reusa **la misma función que ya usa tu upload manual** — así "subido a
   mano" y "importado del hub" terminan en el mismo lugar sin duplicar
   lógica:
   ```python
   @app.post("/api/servers/<server_id>/import-from-hub")
   def import_from_hub(server_id):
       body = request.get_json()
       from model_hub import ModelHubClient
       with tempfile.TemporaryDirectory() as tmp:
           path = ModelHubClient().download(
               body["model_id"], version=body.get("version", "latest"),
               format=body.get("format", "pt"), dest_dir=tmp,
           )
           return _tu_funcion_existente_de_subida(server_id, path.name, path.read_bytes())
   ```
5. **Frontend**: un botón junto al de "Subir modelo" que abre un diálogo con
   2-3 selects encadenados (modelo → versión → formato, poblados desde el
   endpoint del paso 3) y un botón "Importar" que llama al endpoint del
   paso 4. Sin subida de archivo, sin manejo de FormData — la única
   diferencia con el flujo manual es el origen de los bytes.

## Agregar un artifact a una versión ya publicada

Caso típico: publicaste `precintos-s.pt` y más tarde generaste el `.onnx` o
un `.dlc` que en su momento no tenías. **No hace falta republicar** (eso
crearía una versión nueva innecesaria) — `add_artifact()` sube el archivo al
mismo Release ya existente y actualiza `catalog.json` in-place:

```python
from model_hub import ModelHubPublisher

pub = ModelHubPublisher()
pub.add_artifact("yolo-precintos-s-pt", "20260715-145245", "onnx", "/path/precintos-s.onnx")
# ó "dlc", etc. — fmt es libre, no hay una lista cerrada de formatos válidos
```

Si ya existe un artifact con ese mismo `fmt` en esa versión, se reemplaza
(no se duplica). Requiere `MODEL_HUB_GH_TOKEN` (es una operación de
publicador, no de consumidor).

## Uso — publicador (subir un modelo entrenado)

```python
from model_hub import ModelHubPublisher

pub = ModelHubPublisher()
result = pub.publish(
    family="lprocr",
    size="x",
    display_name="LPR OCR v2 — talla X",
    description="Entrenamiento completo, dataset ocr1 + resumes",
    files={"pt": "/path/best.pt", "onnx": "/path/best.onnx"},
    headline_metric={"label": "plate_acc", "value": 0.9962, "format": "percent"},
    chart_series=[{"key": "plate_acc", "label": "plate_acc"}, {"key": "cer", "label": "CER"}],
    rows=[{"epoch": 24, "plate_acc": 0.9954, "cer": 0.0017}, "..."],
    extra_metrics=[{"label": "best_epoch", "value": "92"}],
    version_scheme="v2",
    tags=["patentes"],  # opcional, texto libre — no participa de la identidad
)
print(result["hub_url"])
```

`display_name` determina la identidad del modelo (`model_id = family + slug(display_name)`
— ver `catalog.make_model_id`): publicar dos veces con el **mismo** nombre agrega
una versión nueva a la misma entrada del catálogo (`versions[0]` = la más
reciente) en vez de crear una entrada duplicada. `rows` se sube como asset
propio (no queda inline en `catalog.json`, que así se mantiene liviano
siempre) — `client.get_rows(version)` lo resuelve de forma transparente.

## Formato de cifrado

Ver [`ENCRYPTION.md`](https://github.com/OneclickEB/models-hub/blob/main/ENCRYPTION.md)
en el repo del hub. Implementación de referencia: `crypto.py` en este paquete.

## Integración a neuralVISION

_No implementado — esto es solo el mapa para cuando se decida ejecutarlo
(ver también `neural-vision/tmp/ideas/idea-model-hub-integracion.md`)._

Puntos de enganche identificados en `neural-vision/nv/backend/src/`:

- **`routes/models_api.py`**: expone hoy `list_models()` / `upload_model()` /
  `delete_model()` sobre el directorio local de modelos (`_get_models_path()`).
  Sería el lugar natural para agregar un endpoint nuevo, ej.
  `GET /api/models/hub` (proxy de `ModelHubClient.list_models()`) y
  `POST /api/models/hub/<model_id>/download` (llama a `client.download(...)`
  y deja el archivo desencriptado en `_get_models_path()`, mismo directorio
  que ya usan los modelos locales — así queda utilizable por el resto del
  sistema sin tocar nada más).
- **`plugins/common/plugin_model.py`**: es donde los plugins resuelven qué
  modelo usar (backends CPU/GPU/QNN/HIS) — un modelo bajado del hub debería
  terminar siendo indistinguible de uno subido a mano una vez desencriptado
  en disco, así que no debería requerir cambios acá si el endpoint nuevo
  reusa `_get_models_path()`.
- **Copia del driver**: mismo patrón que `ocrkit`/`model_hub` ya usan en
  `ai-ocr-2026` — `COPY componentes/tools/model-hub-driver/model_hub` al
  Dockerfile de `nv-backend`, agregar `cryptography` a sus requirements.
- **Config**: `MODEL_HUB_KEY_FILE` + `MODEL_HUB_GH_TOKEN` (solo si también va
  a publicar, no solo consumir) en la infra de `nv-backend` — mismo patrón de
  secreto montado como archivo que usan `ai-ocr-2026/ocr-labeling` y
  `yolo11-backend` (evita el problema de passphrases con caracteres
  especiales rotas por interpolación de variables de shell/compose).
- **UI**: `menu:view:models` del SmartClient necesitaría un botón "Importar
  del Model Hub" que liste `client.list_models()` y dispare la descarga —
  no se investigó el componente frontend exacto, queda para cuando se
  ejecute esta fase.
