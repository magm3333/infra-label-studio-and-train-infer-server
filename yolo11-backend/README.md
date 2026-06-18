# YOLO 11 Label Studio ML Backend

Backend simple para inferencia de deteccion YOLO desde Label Studio.

## Arranque

La imagen contiene dependencias pesadas de YOLO/PyTorch. Construyala solo la primera vez, o cuando cambie `requirements.txt`:

```bash
docker build -t yolo11-label-studio-backend:latest yolo11-backend
```

Despues puede levantar normalmente. Aunque use `--build`, Compose no reconstruye esta imagen porque el servicio usa `image` sin `build`:

```bash
docker compose up -d --build
```

El archivo `app.py` se monta como volumen dentro del contenedor, por lo que cambios en el backend solo requieren reiniciar el servicio:

```bash
docker compose restart yolo11-backend
```

El backend queda disponible en:

- Desde Label Studio/Docker: `http://yolo11-backend:9090`
- Desde el host: `http://localhost:9090`

## Configuracion del proyecto en Label Studio

El proyecto debe usar `RectangleLabels` sobre una imagen. Ejemplo minimo:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="precinto"/>
  </RectangleLabels>
</View>
```

Los nombres de los labels deben coincidir con las clases del modelo YOLO. Si el modelo tiene otras clases, use esos nombres exactos en Label Studio.

## Conectar el backend

1. Entrar al proyecto en Label Studio.
2. Ir a `Settings` -> `Model` o `Machine Learning`.
3. Agregar backend con URL: `http://yolo11-backend:9090`.
4. Activar pre-annotations si quiere que Label Studio pida predicciones automaticamente.
5. Guardar y probar con `Validate and Save`.

## Inferencia

Al abrir una tarea, Label Studio puede pedir predicciones al backend y mostrarlas como pre-anotaciones. Tambien se puede usar la accion de recuperar predicciones desde la pantalla de Data Manager si esta disponible en su configuracion.

## Reentrenamiento automatico

El endpoint `/train` puede exportar las anotaciones del proyecto desde la API de Label Studio, convertirlas a formato YOLO y entrenar un nuevo modelo.

Para habilitarlo, configure un token de API en el entorno antes de levantar el compose:

```bash
export LABEL_STUDIO_API_KEY="su_token_de_label_studio"
docker compose up -d --build
```

Luego cree un webhook en Label Studio:

1. Ir a `Organization` o `Project Settings` -> `Webhooks`.
2. URL: `http://yolo11-backend:9090/train`.
3. Eventos sugeridos: `ANNOTATION_CREATED`, `ANNOTATION_UPDATED`.
4. Enviar payload JSON completo.

Cuando llegue un webhook, el backend lanza el entrenamiento en segundo plano. El estado se consulta en:

```bash
curl http://localhost:9090/status
```

Los pesos entrenados se guardan en `./mydata/yolo-backend/models/` y el backend empieza a usar el ultimo `best.pt` si el entrenamiento termina correctamente.

## Variables utiles

- `MODEL_PATH`: modelo inicial. Por defecto `/models/precintos-s.pt`.
- `CONFIDENCE_THRESHOLD`: umbral de confianza. Por defecto `0.25`.
- `TRAIN_EPOCHS`: epocas de entrenamiento. Por defecto `50`.
- `TRAIN_IMGSZ`: tamano de imagen. Por defecto `640`.
- `TRAIN_BATCH`: batch size. Por defecto `8`.
- `LABEL_STUDIO_API_KEY`: token requerido para reentrenamiento automatico.
