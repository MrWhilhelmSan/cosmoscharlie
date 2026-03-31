# Cosmos Charlie (VisioCharlie)

Analista forense visual con **Eagle 2.5** (`nvidia/Eagle2.5-8B`): interfaz **Gradio** para describir imágenes y analizar documentos (PDF vía PyMuPDF).

## Requisitos

- Python 3.10+ recomendado.
- **GPU NVIDIA** con CUDA para rendimiento razonable (el proyecto está pensado para torch con CUDA; en CPU puedes forzarlo, ver abajo).
- Espacio en disco: el modelo se descarga desde Hugging Face la primera vez (no se incluye `eagle_model/` ni el entorno en el repo).

## Instalación

```bash
git clone https://github.com/MrWhilhelmSan/cosmoscharlie.git
cd cosmoscharlie
python3 -m venv vision_env
source vision_env/bin/activate   # Windows: vision_env\Scripts\activate
pip install -r requirements.txt
```

Si usas PyTorch nightly para una GPU muy nueva (p. ej. sm_120), sigue las indicaciones de comentarios en `requirements.txt` para `torch` / `torchvision`.

**Flash Attention 2** es opcional según tu entorno; si falla el arranque, prueba los modos de atención indicados más abajo.

## Uso

```bash
source vision_env/bin/activate
python visiocharlie.py
```

La interfaz queda en **http://0.0.0.0:7860** (accesible en la máquina local como `http://127.0.0.1:7860`).

## Variables de entorno útiles

| Variable | Efecto |
|----------|--------|
| `CUDA_LAUNCH_BLOCKING=1` | Depuración CUDA (muestra mejor el kernel que falla). |
| `VISIOCHARLIE_ATTN=eager` | Atención eager (más compatible; sin Flash Attention 2 / SDPA agresivo). |
| `VISIOCHARLIE_DEVICE=cpu` | Forzar CPU como último recurso. |
| `VISIOCHARLIE_OUTPUT_DIR=/ruta` | Carpeta de respaldo de salidas de documento (por defecto `./visiocharlie_outputs`). |

## Estructura del repo

- `visiocharlie.py` — aplicación principal.
- `requirements.txt` — dependencias Python.
- `checkbase.py`, `test_gpu.py`, `testgpu.py` — utilidades / pruebas locales.

Los directorios `vision_env/` y `eagle_model/` están en `.gitignore` (muy pesados); el modelo se obtiene de Hugging Face al ejecutar.

## Licencia

Revisa las licencias de **NVIDIA Eagle 2.5** y de las dependencias que instales (`transformers`, `torch`, etc.).
