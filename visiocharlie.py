#!/usr/bin/env python3
"""
visiocharlie.py - Analista Forense Visual de Alta Precisión
Modelo: nvidia/Eagle2.5-8B | RTX 5070 Ti (sm_120) | GPU
Entorno: vision_env

Dependencias: torch (nightly cu128 con sm_120), transformers, gradio, Pillow, pymupdf (PDF→imagen), flash-attn.
Ejecutar: python visiocharlie.py

Modo debug (muestra kernel exacto que falla):
  CUDA_LAUNCH_BLOCKING=1 python visiocharlie.py
Forzar atención eager (más compatible, sin flash-attn ni SDPA):
  VISIOCHARLIE_ATTN=eager python visiocharlie.py
Forzar CPU como último recurso:
  VISIOCHARLIE_DEVICE=cpu python visiocharlie.py
Carpeta de respaldo de análisis de documento (por defecto ./visiocharlie_outputs):
  VISIOCHARLIE_OUTPUT_DIR=/ruta/absoluta python visiocharlie.py
"""

import gc
import os
import re
import sys
import time
import traceback
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Configuración de hardware extrema
# ---------------------------------------------------------------------------
MODEL_ID = "nvidia/Eagle2.5-8B"
DTYPE = torch.bfloat16

# Se resuelven al arranque
DEVICE = "cuda"
ATTN_IMPLEMENTATION = "flash_attention_2"
DEVICE_LABEL = "GPU (Flash Attention 2)"

SYSTEM_ROLE = (
    "You are a Digital Pixel Reconstruction Specialist. "
    "Your only job is to produce an exhaustive, obsessively detailed visual inventory of images. "
    "You NEVER summarize, NEVER skip details, and NEVER refuse to describe any image. "
    "Your descriptions must be precise enough that another AI could reconstruct the image pixel by pixel. "
    "You treat every nail, crack, petal, feather, shadow and reflection as critical data."
)

USER_PROMPT = (
    "Analyze this image as a pixel-level reconstruction report. "
    "Structure your response in these sections:\n\n"

    "**QUADRANT ANALYSIS** — Divide the image into 4 quadrants "
    "(top-left, top-right, bottom-left, bottom-right). "
    "For each quadrant list every visible object with: "
    "exact color description (not just 'red' — say 'vibrant crimson red' or 'muted olive green'), "
    "precise texture (smooth, rough, glossy, matte, grainy, fibrous…), "
    "and exact position within the quadrant.\n\n"

    "**EXACT ELEMENT COUNT** — Count every individual element: "
    "petals per flower, feathers per wing, boards in a fence, nails/bolts visible, "
    "leaves on branches, clouds in sky, etc. Give exact numbers, not approximations.\n\n"

    "**MICRO-DETAILS** — Report small marks that are easy to miss: "
    "nail heads, wood grain cracks, paint chips, fur/feather color gradients, "
    "specific petal shapes (rounded tip, pointed, notched), animal poses and expressions, "
    "multicolor patterns on feathers or petals.\n\n"

    "**LIGHTING & COMPOSITION** — Light source direction, shadows (exact position and shape), "
    "reflections, depth of field, and any background gradient or texture.\n\n"

    "Be obsessively thorough. A single missed nail or misidentified petal color counts as failure."
)

# Documentos genéricos: texto + conformación + campos, para alimentar otro LLM con reglas
DOCUMENT_SYSTEM_ROLE = (
    "You are an expert document analyst for any document type (IDs, certificates, diplomas, invoices, contracts, forms). "
    "Your task is MAXIMUM extraction: every readable string, full layout/conformation, and structured fields. "
    "Another LLM will use your output to infer validation rules — be exhaustive and literal. "
    "Follow the exact delimiter format. Use Spanish for narrative inside sections unless the document text is in another language "
    "(transcribe document text verbatim)."
)

DOCUMENT_USER_PROMPT = (
    "Analiza este documento (cualquier tipo: cédula, certificado, diploma, factura, contrato, formulario, etc.).\n\n"
    "Responde EXACTAMENTE con CUATRO bloques y estos delimitadores literales (sin texto antes del primero):\n\n"
    "<<<CONFORMACION>>>\n"
    "Describe la CONFORMACIÓN física y el diseño, independiente del tipo de documento:\n"
    "- Orientación (vertical/horizontal), proporción aproximada, márgenes, rejilla o columnas.\n"
    "- Regiones visibles (cabecera, pie, cuerpo, columnas, bandas, marcos, cajas, tablas, sellos, logos, fotos, firmas).\n"
    "- Orden de lectura sugerido y jerarquía visual (títulos vs cuerpo vs notas).\n"
    "Colores dominantes y secundarios (nombres; hex/rgb si puedes estimarlos), fondos (liso, degradado, patrón, textura).\n"
    "Tipografía aproximada por zona (serif/sans/monoespaciada, tamaño relativo, negritas/cursivas).\n"
    "Líneas, bordes, sombras, marcas de agua, elementos de seguridad visibles (sin afirmar autenticidad).\n"
    "NO transcribas aquí párrafos completos de texto del documento; solo referencias breves de estilo si ayudan.\n\n"
    "<<<TEXTO_INTEGRAL>>>\n"
    "Transcripción lo más COMPLETA posible de TODO texto legible, en orden de lectura natural (arriba→abajo, izquierda→derecha). "
    "Incluye títulos, subtítulos, etiquetas, números, fechas, códigos, pies de página, sellos con texto, URLs, emails. "
    "Si hay tablas, indica filas/columnas de forma clara. Si algo es ilegible, marca [ilegible]. "
    "Si el documento no es en español, transcribe en el idioma original.\n\n"
    "<<<DATOS_CAMPOS>>>\n"
    "Lista estructurada para reglas / validación (viñetas). Para cada elemento indica cuando aplique:\n"
    "- **Nombre del campo o etiqueta** (o «texto libre» si no hay etiqueta)\n"
    "- **Valor** (exacto como en el documento)\n"
    "- **Tipo inferido** (texto, número, fecha, moneda, código, firma, foto, sello, etc.)\n"
    "- **Ubicación** (zona: ej. cabecera central, tabla fila 2, bloque inferior derecho)\n"
    "- **Formato observado** (ej. dd/mm/aaaa, NNN.NNN.NNN, mayúsculas)\n"
    "Incluye también relaciones obvias (ej. «expedido por», «válido hasta») si se deducen del layout.\n\n"
    "<<<FIN>>>\n"
)

# ---------------------------------------------------------------------------
# Detección de GPU soportada (evita cudaErrorNoKernelImageForDevice en Blackwell)
# ---------------------------------------------------------------------------
def _flash_attn_supports_current_gpu() -> bool:
    """
    Verifica si flash_attn está instalado Y soporta la GPU actual.
    Hace una llamada real a flash_attn_gpu.fwd con un tensor diminuto.
    """
    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_func
        q = torch.randn(1, 1, 4, 16, dtype=torch.bfloat16, device="cuda")
        flash_attn_func(q, q, q)
        return True
    except Exception:
        return False


def resolve_device_and_attention():
    """
    Orden de preferencia:
      1. cuda + flash_attention_2  — si flash_attn está instalado y soporta la GPU
      2. cuda + sdpa               — si flash_attn falla o no está (SDPA de PyTorch)
      3. cpu  + eager              — último recurso

    Variables de entorno opcionales:
      VISIOCHARLIE_ATTN=sdpa|eager|flash_attention_2  → fuerza un modo concreto
      VISIOCHARLIE_DEVICE=cpu                          → fuerza CPU
    """
    global DEVICE, ATTN_IMPLEMENTATION, DEVICE_LABEL

    if os.environ.get("VISIOCHARLIE_DEVICE", "").strip().lower() == "cpu":
        DEVICE = "cpu"
        ATTN_IMPLEMENTATION = "eager"
        DEVICE_LABEL = "CPU (eager)"
        print("VISIOCHARLIE_DEVICE=cpu: modo CPU forzado.")
        return

    if not torch.cuda.is_available():
        DEVICE = "cpu"
        ATTN_IMPLEMENTATION = "eager"
        DEVICE_LABEL = "CPU (eager)"
        print("CUDA no disponible. Usando CPU.")
        return

    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    gpu_name = torch.cuda.get_device_name(0)

    if arch not in torch.cuda.get_arch_list():
        DEVICE = "cpu"
        ATTN_IMPLEMENTATION = "eager"
        DEVICE_LABEL = f"CPU (eager) — {gpu_name} ({arch}) no soportada por este PyTorch"
        print(f"{gpu_name} ({arch}) no soportada. Usando CPU.")
        return

    DEVICE = "cuda"

    # Respetar override manual
    attn_env = os.environ.get("VISIOCHARLIE_ATTN", "").strip().lower()
    if attn_env in ("sdpa", "eager", "flash_attention_2"):
        ATTN_IMPLEMENTATION = attn_env
        DEVICE_LABEL = f"GPU {gpu_name} ({attn_env}) [manual]"
        print(f"Atención forzada por VISIOCHARLIE_ATTN={attn_env}.")
        return

    # Auto-detección: probar flash_attn con la GPU actual
    if _flash_attn_supports_current_gpu():
        ATTN_IMPLEMENTATION = "flash_attention_2"
        DEVICE_LABEL = f"GPU {gpu_name} (Flash Attention 2)"
        print(f"{gpu_name}: flash_attn soportada. Usando Flash Attention 2 + bfloat16.")
    else:
        # flash_attn no soporta esta GPU (p. ej. sm_120 sin recompilar con TORCH_CUDA_ARCH_LIST=12.0)
        # SDPA de PyTorch funciona correctamente con sm_120 (verificado)
        ATTN_IMPLEMENTATION = "sdpa"
        DEVICE_LABEL = f"GPU {gpu_name} (SDPA)"
        print(
            f"{gpu_name}: flash_attn no soporta sm_120 en este build. "
            "Usando SDPA (PyTorch nativo). "
            "Para recuperar Flash Attention 2: "
            "pip uninstall flash-attn && "
            "TORCH_CUDA_ARCH_LIST=\"8.0;8.6;9.0;10.0;12.0\" pip install flash-attn --no-build-isolation"
        )


# ---------------------------------------------------------------------------
# Carga del modelo (una sola vez al iniciar)
# ---------------------------------------------------------------------------
def _patch_eagle_vision_attn():
    """
    Eagle2.5-8B hardcodea 'flash_attention_2' para el encoder SigLIP en su __init__.
    Este patch hace que respete el attn_implementation global en lugar de forzarlo.
    Se aplica antes de importar el modelo para sobrevivir limpiezas de caché.
    """
    try:
        import importlib
        import sys
        module_name = None
        for key in sys.modules:
            if "modeling_eagle2_5_vl" in key:
                module_name = key
                break
        if module_name:
            mod = sys.modules[module_name]
            original_init = mod.Eagle2_5_VLForConditionalGeneration.__init__

            def patched_init(self, config, vision_model=None, language_model=None):
                # Forzar attn_implementation correcto en el vision config antes de que
                # el __init__ original lo sobreescriba con 'flash_attention_2'
                if hasattr(config, 'vision_config'):
                    config.vision_config._attn_implementation = (
                        getattr(config, '_attn_implementation_internal', None) or ATTN_IMPLEMENTATION
                    )
                original_init(self, config, vision_model=vision_model, language_model=language_model)

            mod.Eagle2_5_VLForConditionalGeneration.__init__ = patched_init
    except Exception as e:
        print(f"[patch eagle] {e} — continuando sin patch (el archivo caché ya fue modificado)")


def load_model_and_processor():
    """Carga Eagle2.5-8B con bfloat16 en GPU usando ATTN_IMPLEMENTATION resuelto al arranque."""
    from transformers import AutoConfig
    print(f"Cargando modelo y processor... ({DEVICE}, {ATTN_IMPLEMENTATION})")

    # Cargar config y forzar la implementación en todos los sub-configs ANTES de
    # instanciar el modelo. Eagle2.5-8B hardcodea flash_attention_2 en tres sitios;
    # esto garantiza que se use ATTN_IMPLEMENTATION en todos ellos.
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config._attn_implementation_internal = ATTN_IMPLEMENTATION
    config._attn_implementation_autoset = False
    if hasattr(config, "vision_config"):
        config.vision_config._attn_implementation_internal = ATTN_IMPLEMENTATION
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation_internal = ATTN_IMPLEMENTATION

    model = AutoModel.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=DTYPE,
    )
    model = model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=True,
    )
    processor.tokenizer.padding_side = "left"

    return model, processor


# Carga global (se reutiliza entre análisis)
_model = None
_processor = None


def get_model_and_processor():
    global _model, _processor
    if _model is None or _processor is None:
        _model, _processor = load_model_and_processor()
    return _model, _processor


# ---------------------------------------------------------------------------
# Limpieza de memoria GPU
# ---------------------------------------------------------------------------
def clear_gpu_memory():
    """Libera memoria de la GPU (cache de CUDA y recolector de Python)."""
    global _model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # No descargamos el modelo de la GPU para no tener que recargarlo;
    # solo liberamos caché y objetos temporales.
    return "Memoria GPU liberada (cache vaciado)."


# ---------------------------------------------------------------------------
# Inferencia forense
# ---------------------------------------------------------------------------
_RE_CONFORMACION = re.compile(r"<<<\s*CONFORMACION\s*>>>", re.IGNORECASE)
_RE_TEXTO_INTEGRAL = re.compile(r"<<<\s*TEXTO_INTEGRAL\s*>>>", re.IGNORECASE)
_RE_DATOS_CAMPOS = re.compile(r"<<<\s*DATOS_CAMPOS\s*>>>", re.IGNORECASE)
_RE_FIN_BLOQUE = re.compile(r"<<<\s*FIN\s*>>>", re.IGNORECASE)
# Formato antiguo (compatibilidad)
_RE_PARTE_GRAFICA = re.compile(r"<<<\s*PARTE_GRAFICA\s*>>>", re.IGNORECASE)


def sanitize_gradio_text(value: str | None) -> str:
    """Evita bytes nulos y caracteres de control que rompen JSON/WebSocket en Gradio."""
    if value is None:
        return ""
    s = str(value).replace("\x00", "")
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if ch in "\n\t\r":
            out.append(ch)
        elif o >= 32:
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def pdf_to_pil_images(
    path: str | Path,
    *,
    dpi: float = 144.0,
    all_pages: bool = False,
    max_pages: int = 30,
    max_side_px: int = 2048,
) -> tuple[list[Image.Image] | None, str]:
    """
    Renderiza PDF a imágenes RGB (PyMuPDF). Si all_pages es False, solo la primera página.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, (
            "[Error] Falta PyMuPDF. Instala con: pip install pymupdf"
        )

    path = Path(path)
    if not path.is_file():
        return None, f"[Error] No existe el archivo: {path}"

    note_parts: list[str] = []
    try:
        doc = fitz.open(path)
    except Exception as e:
        return None, f"[Error] No se pudo abrir el PDF: {e}"

    n = len(doc)
    if n == 0:
        doc.close()
        return None, "[Error] PDF sin páginas."

    count = n if all_pages else 1
    count = min(count, max_pages, n)
    if n > max_pages and all_pages:
        note_parts.append(
            f"(Solo se procesan las primeras {max_pages} páginas de {n}.)"
        )

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []

    for i in range(count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        im = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        im = im.convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m > max_side_px:
            s = max_side_px / m
            im = im.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
        images.append(im)

    doc.close()
    note = " ".join(note_parts).strip()
    return images, note


def load_image_or_pdf_to_pil_list(
    image_source,
    *,
    pdf_path: str | None,
    all_pdf_pages: bool,
    pdf_dpi: float = 144.0,
) -> tuple[list[Image.Image] | None, str]:
    """
    Prioridad: si hay `pdf_path` válido, se usa el PDF; si no, `image_source` (ruta imagen o array).
    Una ruta .png/.jpg en image_source se carga como una sola imagen.
    """
    if pdf_path:
        p = Path(str(pdf_path))
        if p.is_file() and p.suffix.lower() == ".pdf":
            return pdf_to_pil_images(
                p, dpi=pdf_dpi, all_pages=all_pdf_pages
            )
        return None, "[Error] El PDF no es un archivo válido."

    if image_source is None:
        return None, "[Error] Sube una imagen o un PDF."

    if isinstance(image_source, Image.Image):
        return [image_source.convert("RGB")], ""

    if isinstance(image_source, str):
        ip = Path(image_source)
        if not ip.is_file():
            return None, f"[Error] No existe el archivo: {image_source}"
        if ip.suffix.lower() == ".pdf":
            return pdf_to_pil_images(
                ip, dpi=pdf_dpi, all_pages=all_pdf_pages
            )
        try:
            return [Image.open(ip).convert("RGB")], ""
        except Exception as e:
            return None, f"[Error] No se pudo abrir la imagen: {e}"

    try:
        return [Image.fromarray(image_source).convert("RGB")], ""
    except Exception as e:
        return None, f"[Error] Entrada de imagen no válida: {e}"


def _slice_between_markers(
    raw_text: str,
    start_re: re.Pattern[str],
    end_re: re.Pattern[str],
    *,
    search_from: int = 0,
) -> str | None:
    m_s = start_re.search(raw_text, search_from)
    if not m_s:
        return None
    start = m_s.end()
    m_e = end_re.search(raw_text, start)
    end = m_e.start() if m_e else len(raw_text)
    return raw_text[start:end].strip()


def split_document_output(raw_text: str) -> tuple[str, str, str]:
    """
    Separa salida del modo documento en:
    (conformación/diseño, texto integral, datos estructurados).

    Formato nuevo: <<<CONFORMACION>>> <<<TEXTO_INTEGRAL>>> <<<DATOS_CAMPOS>>> <<<FIN>>>
    Formato legado: <<<PARTE_GRAFICA>>> … <<<DATOS_CAMPOS>>> (sin texto integral dedicado).
    """
    if not raw_text or raw_text.startswith("[Error"):
        return sanitize_gradio_text(raw_text or ""), "", ""

    has_new = bool(_RE_CONFORMACION.search(raw_text) or _RE_TEXTO_INTEGRAL.search(raw_text))

    if has_new:
        conform = _slice_between_markers(raw_text, _RE_CONFORMACION, _RE_TEXTO_INTEGRAL)
        texto_int = _slice_between_markers(raw_text, _RE_TEXTO_INTEGRAL, _RE_DATOS_CAMPOS)
        datos = _slice_between_markers(raw_text, _RE_DATOS_CAMPOS, _RE_FIN_BLOQUE)
        conform = conform if conform is not None else ""
        texto_int = texto_int if texto_int is not None else ""
        datos = datos if datos is not None else ""
    else:
        m_g = _RE_PARTE_GRAFICA.search(raw_text)
        if m_g:
            after = m_g.end()
            m_d = _RE_DATOS_CAMPOS.search(raw_text, after)
            if m_d:
                conform = raw_text[after : m_d.start()].strip()
                after_d = m_d.end()
                m_f = _RE_FIN_BLOQUE.search(raw_text, after_d)
                fin = m_f.start() if m_f else len(raw_text)
                datos = raw_text[after_d:fin].strip()
                texto_int = ""
            else:
                conform = raw_text[after:].strip()
                datos = ""
                texto_int = ""
        elif _RE_DATOS_CAMPOS.search(raw_text):
            conform = ""
            texto_int = ""
            datos = _slice_between_markers(raw_text, _RE_DATOS_CAMPOS, _RE_FIN_BLOQUE) or ""
        else:
            return (
                sanitize_gradio_text(raw_text),
                "",
                sanitize_gradio_text(
                    "(No se encontraron delimitadores esperados. Revisa la salida completa del modelo.)"
                ),
            )

    if not (conform or "").strip():
        conform = (
            "(El bloque de conformación quedó vacío o el modelo no usó <<<CONFORMACION>>>. "
            "Revisa la salida completa.)"
        )

    return (
        sanitize_gradio_text(conform),
        sanitize_gradio_text(texto_int),
        sanitize_gradio_text(datos),
    )


def _parse_time_seconds(tm: str | None) -> float:
    if not tm:
        return 0.0
    m = re.search(r"([\d.]+)\s*s", str(tm), re.I)
    if m:
        return float(m.group(1))
    m2 = re.search(r"([\d.]+)", str(tm))
    return float(m2.group(1)) if m2 else 0.0


def _document_output_dir() -> Path:
    env = os.environ.get("VISIOCHARLIE_OUTPUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent / "visiocharlie_outputs").resolve()


def save_document_analysis_artifact(
    graphic: str,
    texto_integral: str,
    datos: str,
    raw_full: str,
    *,
    extra_note: str = "",
) -> str:
    """
    Escribe el resultado en disco para que no se pierda si el navegador se refresca
    o se corta la conexión SSE (análisis largos).
    """
    out_dir = _document_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{ts}_documento.txt"
    header_note = extra_note.strip()
    content = (
        f"VisioCharlie — exportación de análisis de documento\n"
        f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    if header_note:
        content += f"{header_note}\n"
    content += (
        f"\n===== CONFORMACIÓN Y DISEÑO =====\n{graphic}\n\n"
        f"===== TEXTO INTEGRAL =====\n{texto_integral}\n\n"
        f"===== DATOS / CAMPOS =====\n{datos}\n\n"
        f"===== SALIDA CRUDA DEL MODELO =====\n{raw_full}\n"
    )
    path.write_text(content, encoding="utf-8")
    print(
        f"[VisioCharlie] Análisis de documento guardado en: {path}",
        file=sys.stderr,
        flush=True,
    )
    return str(path)


def analyze_image(
    image_source,
    clear_after: bool = False,
    mode: str = "forensic",
):
    """
    Ejecuta el análisis sobre la imagen.
    image_source: path (str) o numpy array (Gradio suele pasar numpy desde la cámara/upload).
    mode: 'forensic' | 'document'
    """
    if image_source is None:
        return (
            "[Error] No se ha subido ninguna imagen.",
            None,
            None,
        )

    model, processor = get_model_and_processor()

    # Normalizar entrada: PIL, ruta, o numpy (Gradio)
    if isinstance(image_source, Image.Image):
        pil_image = image_source.convert("RGB")
    elif isinstance(image_source, str):
        image_path = Path(image_source)
        if not image_path.exists():
            return (
                f"[Error] No existe el archivo: {image_source}",
                None,
                None,
            )
        if image_path.suffix.lower() == ".pdf":
            imgs, err = pdf_to_pil_images(image_path, all_pages=False)
            if not imgs:
                return err or "[Error] PDF vacío.", None, None
            pil_image = imgs[0]
        else:
            pil_image = Image.open(image_path).convert("RGB")
    else:
        pil_image = Image.fromarray(image_source).convert("RGB")

    if mode == "document":
        sys_role, user_prompt = DOCUMENT_SYSTEM_ROLE, DOCUMENT_USER_PROMPT
    else:
        sys_role, user_prompt = SYSTEM_ROLE, USER_PROMPT

    messages = [
        {
            "role": "system",
            "content": sys_role,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    try:
        text_list = [
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        ]
        image_inputs, video_inputs = processor.process_vision_info(messages)
        inputs = processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(DEVICE)

        # Medir tiempo de inferencia
        start = time.perf_counter()
        max_tokens = 4096 if mode == "document" else 2048
        temp = 0.1 if mode == "document" else 0.15
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                repetition_penalty=1.05,
            )
        elapsed = time.perf_counter() - start

        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result_text = output_text[0].strip() if output_text else ""
        result_text = sanitize_gradio_text(result_text)

        if clear_after:
            clear_gpu_memory()

        time_msg = f"{elapsed:.3f} s"
        return result_text, time_msg, f"Tiempo de inferencia: {time_msg}"

    except Exception as e:
        return (
            f"[Error durante el análisis] {type(e).__name__}: {e}",
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Generación de documento sintético (POC visual)
# ---------------------------------------------------------------------------
def _header_color_from_description(graphic_text: str) -> tuple[int, int, int]:
    """Intenta extraer un color de cabecera desde la descripción gráfica del modelo."""
    if not graphic_text:
        return (12, 74, 138)
    m = re.search(r"#([0-9A-Fa-f]{6})\b", graphic_text)
    if m:
        hx = m.group(1)
        return tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    m = re.search(
        r"rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)",
        graphic_text,
        re.I,
    )
    if m:
        return tuple(min(255, int(x)) for x in m.groups())  # type: ignore[return-value]
    return (12, 74, 138)


# Paleta POC inspirada en descripciones típicas de cédula CO (sin copiar documento real)
_CO_BLUE = (12, 74, 138)
_CO_YELLOW = (245, 210, 66)
_CO_ORANGE = (232, 120, 40)
_CO_RED = (193, 30, 40)
_CO_BEIGE_TOP = (238, 232, 218)
_CO_BEIGE_BOT = (220, 214, 198)
_CO_MINT = (190, 220, 200)


def _format_colombian_cedula_display(num: str) -> str:
    """Formatea dígitos estilo cédula CO: 1.106.512.640 (grupos de 3 desde la derecha)."""
    s = (num or "").strip()
    if not s:
        return "—"
    digits = re.sub(r"\D", "", s)
    if not digits:
        return "—"
    if len(digits) < 6:
        return digits
    parts: list[str] = []
    rest = digits
    while rest:
        if len(rest) <= 3:
            parts.insert(0, rest)
            break
        parts.insert(0, rest[-3:])
        rest = rest[:-3]
    return ".".join(parts)


def _blend_rgb(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _draw_gradient_bg(img: Image.Image, top: tuple[int, int, int], bot: tuple[int, int, int]) -> None:
    px = img.load()
    w, h = img.size
    for y in range(h):
        t = y / max(h - 1, 1)
        r, g, b = _blend_rgb(top, bot, t)
        for x in range(w):
            px[x, y] = (r, g, b)


def _draw_circle_pattern(
    draw: ImageDraw.ImageDraw,
    x_max: int,
    y_bottom: int,
    y_top: int,
    margin: int = 10,
) -> None:
    """Patrón decorativo de círculos en la zona izquierda del cuerpo (solo POC)."""
    step = 36
    for cy in range(y_top + margin, y_bottom - margin, step):
        for cx in range(margin, max(x_max, margin + step) - margin, step):
            off = ((cy // step) + (cx // step)) % 3 * 4
            bbox = [cx + off, cy, cx + off + 22, cy + 22]
            draw.ellipse(bbox, outline=(200, 192, 178), width=1)


def _draw_top_geometric_band(draw: ImageDraw.ImageDraw, w: int, h_band: int) -> None:
    """Banda superior con formas geométricas en amarillo / azul / naranja / rojo."""
    strip_h = max(6, h_band // 5)
    y = 0
    colors_cycle = (_CO_YELLOW, _CO_BLUE, _CO_ORANGE, _CO_RED, _CO_YELLOW)
    i = 0
    while y < h_band:
        draw.rectangle([0, y, w, y + strip_h], fill=colors_cycle[i % len(colors_cycle)])
        y += strip_h
        i += 1
    # Triángulos / trapecios simulados en la franja principal
    mid = h_band - 14
    draw.polygon(
        [(0, h_band), (0, mid), (w // 4, h_band)],
        fill=_blend_rgb(_CO_BLUE, (0, 0, 0), 0.15),
    )
    draw.polygon(
        [(w, h_band), (w, mid), (3 * w // 4, h_band)],
        fill=_blend_rgb(_CO_ORANGE, (255, 255, 255), 0.1),
    )


def _draw_right_geometric_band(
    draw: ImageDraw.ImageDraw, w: int, h: int, x0: int, band_w: int, y_start: int
) -> None:
    x1 = w
    y = y_start
    h_rem = h - y_start - 8
    step = 42
    cols = (_CO_YELLOW, _CO_BLUE, _CO_RED, _CO_MINT, _CO_ORANGE)
    i = 0
    while y < h - 8:
        y2 = min(y + step, h - 8)
        draw.rectangle([x0, y, x1, y2], fill=cols[i % len(cols)])
        i += 1
        y = y2


def _draw_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    x: int,
    y: int,
    max_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int = 4,
) -> int:
    """Dibuja líneas ya partidas; devuelve y final."""
    yy = y
    for line in lines:
        draw.text((x, yy), line, fill=fill, font=font)
        try:
            bbox = draw.textbbox((x, yy), line, font=font)
            yy = bbox[3] + line_gap
        except Exception:
            yy += 22 + line_gap
    return yy


def _split_header_lines(titulo: str) -> list[str]:
    t = (titulo or "").strip()
    if not t:
        return [
            "REPÚBLICA DE COLOMBIA",
            "IDENTIFICACIÓN PERSONAL",
            "CÉDULA DE CIUDADANÍA",
        ]
    if "\n" in t:
        return [ln.strip() for ln in t.splitlines() if ln.strip()]
    upper = re.sub(r"\s+", " ", t.upper())
    upper_norm = upper.replace("CIUDADANIA", "CIUDADANÍA").replace("CEDULA", "CÉDULA")
    if "COLOMBIA" in upper_norm:
        return [
            "REPÚBLICA DE COLOMBIA",
            "IDENTIFICACIÓN PERSONAL",
            "CÉDULA DE CIUDADANÍA",
        ]
    return [t]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_dummy_id_card(
    profile: dict | None,
    nombre: str,
    apellidos: str,
    numero_documento: str,
    fecha_nacimiento: str,
    nacionalidad: str,
    titulo_documento: str,
) -> Image.Image:
    """
    Imagen sintética estilo cédula (POC): colores y layout inspirados en descripciones
    típicas (bandas geométricas, degradado, círculos, foto a la derecha). No es un
    documento real ni reproduce seguridad (hologramas, etc.).
    """
    graphic = (profile or {}).get("graphic") or ""
    W, H = 1020, 640
    img = Image.new("RGB", (W, H), _CO_BEIGE_TOP)
    _draw_gradient_bg(img, _CO_BEIGE_TOP, _CO_BEIGE_BOT)
    draw = ImageDraw.Draw(img)

    h_strip = 96
    _draw_top_geometric_band(draw, W, h_strip)

    accent_blue = _header_color_from_description(graphic)
    bar_h = 78
    bar_y = h_strip
    draw.rectangle([0, bar_y, W, bar_y + bar_h], fill=accent_blue)

    font_head_l = _load_font(22, bold=True)
    font_head_s = _load_font(15, bold=False)
    font_label = _load_font(17, bold=True)
    font_val = _load_font(19, bold=False)
    font_small = _load_font(12, bold=False)

    header_lines = _split_header_lines(titulo_documento)
    hx, hy = 24, bar_y + 10
    draw.text((hx, hy), header_lines[0].upper(), fill=(255, 255, 255), font=font_head_l)
    hy += 28
    for extra in header_lines[1:]:
        draw.text((hx, hy), extra.upper(), fill=(245, 240, 230), font=font_head_s)
        hy += 20

    body_top = bar_y + bar_h + 10
    right_band_w = 52
    body_right = W - right_band_w - 8
    photo_w, photo_h = 200, 260
    photo_x = body_right - photo_w - 18
    photo_y = body_top + 6

    _draw_circle_pattern(draw, int(photo_x - 8), H - 50, body_top)

    cedula_fmt = _format_colombian_cedula_display(numero_documento)
    ap = (apellidos or "").strip() or "—"
    nom = (nombre or "").strip() or "—"
    fnac = (fecha_nacimiento or "").strip() or "—"
    nac = (nacionalidad or "").strip() or "—"

    tx = 28
    ty = body_top + 8
    line_skip = 40

    def field(label: str, value: str, big: bool = False) -> None:
        nonlocal ty
        draw.text((tx, ty), label, fill=(55, 52, 48), font=font_label)
        ty += 22
        vf = font_val if big else _load_font(18, bold=False)
        draw.text((tx, ty), value, fill=(18, 18, 18), font=vf)
        ty += line_skip

    field("APELLIDOS", ap)
    field("NOMBRE", nom)
    field("CÉDULA No.", cedula_fmt, big=True)
    if fnac != "—":
        field("FECHA DE NAC.", fnac)
    if nac != "—":
        field("NACIONALIDAD", nac)

    draw.rectangle(
        [photo_x, photo_y, photo_x + photo_w, photo_y + photo_h],
        outline=(90, 90, 88),
        width=2,
        fill=(235, 233, 228),
    )
    draw.text(
        (photo_x + 58, photo_y + photo_h // 2 - 10),
        "FOTO",
        fill=(160, 156, 150),
        font=_load_font(20, bold=True),
    )
    draw.text(
        (photo_x + 12, photo_y + photo_h - 22),
        "(portador)",
        fill=(130, 126, 120),
        font=font_small,
    )

    sig_y = min(ty + 12, H - 78)
    sig_x = tx
    sig_w = min(photo_x - sig_x - 24, 420)
    draw.text((sig_x, sig_y), "FIRMA", fill=(55, 52, 48), font=font_label)
    draw.line(
        [(sig_x, sig_y + 38), (sig_x + sig_w, sig_y + 38)],
        fill=(60, 58, 55),
        width=2,
    )
    scribble_y = sig_y + 32
    for i in range(8):
        dx = i * (sig_w // 8)
        wobble = (i % 3 - 1) * 3
        draw.line(
            [
                (sig_x + dx, scribble_y + wobble),
                (sig_x + dx + 28, scribble_y - wobble),
            ],
            fill=(120, 118, 115),
            width=1,
        )
    draw.text(
        (sig_x, sig_y + 44),
        "(firmante no identificado — demo)",
        fill=(120, 116, 110),
        font=font_small,
    )

    _draw_right_geometric_band(draw, W, H, W - right_band_w, right_band_w, body_top - 4)

    draw.rectangle([0, 0, W, H], outline=accent_blue, width=4)

    disclaimer = (
        "DOCUMENTO SINTÉTICO · SOLO DEMOSTRACIÓN · NO VÁLIDO COMO IDENTIFICACIÓN"
    )
    draw.text((24, H - 28), disclaimer, fill=(130, 125, 118), font=font_small)

    return img


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------
def build_interface():
    with gr.Blocks(title="VisioCharlie - Analista Forense Visual") as demo:
        # Debe crearse dentro del contexto Blocks (Gradio 6); si no, State y triggers fallan.
        doc_profile_state = gr.State(value=None)

        gr.Markdown(
            "# VisioCharlie – Análisis Visual de Ultra-Precisión\n"
            f"**Modelo:** nvidia/Eagle2.5-8B | bfloat16 | **{DEVICE_LABEL}**"
        )

        with gr.Tabs():
            with gr.Tab("Análisis forense"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Sube una imagen",
                            type="filepath",
                            sources=["upload", "clipboard"],
                        )
                        with gr.Row():
                            analyze_btn = gr.Button("Analizar imagen", variant="primary")
                            clear_mem_btn = gr.Button("Limpiar memoria GPU", variant="secondary")

                        clear_after = gr.Checkbox(
                            label="Limpiar memoria GPU después de cada análisis",
                            value=False,
                        )

                    with gr.Column(scale=1):
                        analysis_output = gr.Textbox(
                            label="Análisis forense",
                            lines=24,
                            max_lines=32,
                        )
                        time_output = gr.Textbox(
                            label="Tiempo de inferencia",
                            interactive=False,
                        )
                        time_badge = gr.Markdown(visible=True)

                def run_analysis(img, clear_after_checked):
                    text, time_val, time_md = analyze_image(
                        img, clear_after=clear_after_checked, mode="forensic"
                    )
                    time_str = time_val if time_val else "—"
                    badge = f"**Tiempo de inferencia:** {time_str}" if time_md else ""
                    return text, time_str, badge

                analyze_btn.click(
                    fn=run_analysis,
                    inputs=[image_input, clear_after],
                    outputs=[analysis_output, time_output, time_badge],
                )

                clear_mem_btn.click(
                    fn=lambda: clear_gpu_memory(),
                    inputs=[],
                    outputs=[time_output],
                )

                gr.Markdown(
                    "*Análisis por cuadrantes · color · textura · posición · conteo exhaustivo de elementos*"
                )

            with gr.Tab("Documento (extracción para reglas)"):
                gr.Markdown(
                    "Sube una **imagen** (PNG, JPG, JPEG, WebP) **o un PDF**. "
                    "Si eliges PDF, se convierte a imagen (raster) con PyMuPDF; puedes procesar **solo la primera página** "
                    "o **todas** (hasta 30). "
                    "El modelo separa **conformación y diseño**, **texto integral** y **campos estructurados** — "
                    "ideal para copiar a otro LLM y definir reglas de validación.\n\n"
                    "**Importante:** en análisis largos (varios minutos) **no refresques la pestaña** mientras corre: "
                    "el navegador puede cortar la conexión y perder la respuesta en pantalla. "
                    "Cada análisis **exitoso** se guarda también en la carpeta `visiocharlie_outputs/` "
                    "(ruta completa en el recuadro verde de tiempo). Puedes cambiar la carpeta con la variable de entorno "
                    "`VISIOCHARLIE_OUTPUT_DIR`."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        doc_image = gr.Image(
                            label="Imagen del documento (si no usas PDF)",
                            type="filepath",
                            sources=["upload", "clipboard"],
                        )
                        doc_pdf = gr.File(
                            label="O PDF",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        doc_pdf_pages = gr.Radio(
                            choices=[
                                "Primera página",
                                "Todas las páginas (máx. 30)",
                            ],
                            value="Primera página",
                            label="Páginas del PDF",
                        )
                        doc_pdf_dpi = gr.Slider(
                            minimum=96,
                            maximum=220,
                            value=144,
                            step=8,
                            label="DPI al convertir PDF a imagen",
                        )
                        with gr.Row():
                            doc_analyze_btn = gr.Button(
                                "Analizar documento", variant="primary"
                            )
                        doc_clear_after = gr.Checkbox(
                            label="Limpiar memoria GPU después del análisis",
                            value=False,
                        )

                    with gr.Column(scale=1):
                        doc_graphic = gr.Textbox(
                            label="Conformación y diseño (layout, zonas, colores, tipografía)",
                            lines=10,
                            max_lines=32,
                        )
                        doc_texto_integral = gr.Textbox(
                            label="Texto integral (transcripción ordenada)",
                            lines=10,
                            max_lines=32,
                        )
                        doc_data = gr.Textbox(
                            label="Datos / campos para reglas (etiqueta, valor, ubicación, formato)",
                            lines=10,
                            max_lines=32,
                        )
                        doc_raw = gr.Textbox(
                            label="Salida completa del modelo (todas las páginas concatenadas)",
                            lines=6,
                            max_lines=12,
                        )
                        doc_time = gr.Textbox(
                            label="Tiempo de inferencia",
                            interactive=False,
                        )
                        doc_badge = gr.Markdown()

                def run_doc_analysis(img, pdf_file, pages_mode, pdf_dpi, clear_after_checked):
                    try:
                        all_pages = str(pages_mode).startswith("Todas")
                        pil_list, conv_note = load_image_or_pdf_to_pil_list(
                            img,
                            pdf_path=pdf_file if pdf_file else None,
                            all_pdf_pages=all_pages,
                            pdf_dpi=float(pdf_dpi),
                        )
                        if pil_list is None:
                            msg = conv_note or "[Error] No se pudo cargar el documento."
                            empty_profile: dict = {"graphic": "", "data": "", "texto": ""}
                            return (
                                msg,
                                "",
                                "",
                                msg,
                                "—",
                                "**Error**",
                                empty_profile,
                            )

                        page_texts: list[str] = []
                        g_parts: list[str] = []
                        t_parts: list[str] = []
                        d_parts: list[str] = []
                        total_sec = 0.0
                        for i, pil in enumerate(pil_list):
                            t, tm, time_md = analyze_image(
                                pil, clear_after=False, mode="document"
                            )
                            total_sec += _parse_time_seconds(tm)
                            page_texts.append(
                                f"=== PÁGINA {i + 1} / {len(pil_list)} ===\n{t}"
                            )
                            c, ti, d = split_document_output(t)
                            g_parts.append(
                                f"=== PÁGINA {i + 1} / {len(pil_list)} ===\n{c}"
                            )
                            t_parts.append(
                                f"=== PÁGINA {i + 1} / {len(pil_list)} ===\n{ti}"
                            )
                            d_parts.append(
                                f"=== PÁGINA {i + 1} / {len(pil_list)} ===\n{d}"
                            )

                        if clear_after_checked:
                            clear_gpu_memory()

                        text = "\n\n".join(page_texts)
                        if conv_note:
                            text = conv_note + "\n\n" + text
                        text = sanitize_gradio_text(text)
                        graphic = "\n\n".join(g_parts)
                        texto_integral = "\n\n".join(t_parts)
                        datos = "\n\n".join(d_parts)

                        time_str = f"{total_sec:.3f} s"
                        saved_path = ""
                        try:
                            saved_path = save_document_analysis_artifact(
                                graphic,
                                texto_integral,
                                datos,
                                text,
                                extra_note=conv_note or "",
                            )
                        except OSError as e:
                            print(
                                f"[VisioCharlie] No se pudo guardar el archivo: {e}",
                                file=sys.stderr,
                                flush=True,
                            )
                            saved_path = f"(error al guardar: {e})"

                        badge = f"**Tiempo total:** {time_str}"
                        if conv_note:
                            badge += f" · `{conv_note}`"
                        if saved_path and not saved_path.startswith("("):
                            badge += (
                                f"\n\n**Copia de seguridad en disco:** `{saved_path}` "
                                "(si refrescaste la página y perdiste el texto, ábrelo aquí)."
                            )
                        profile = {
                            "graphic": graphic,
                            "data": datos,
                            "texto": texto_integral,
                        }
                        return (
                            graphic,
                            texto_integral,
                            datos,
                            text,
                            time_str,
                            badge,
                            profile,
                        )
                    except Exception:
                        err = traceback.format_exc()
                        print(
                            "[VisioCharlie] Error en análisis de documento:\n",
                            err,
                            file=sys.stderr,
                            flush=True,
                        )
                        msg = (
                            f"[Error interno al procesar la respuesta]\n\n"
                            f"Detalle (también en la consola del servidor):\n{err}"
                        )
                        empty_profile: dict = {"graphic": "", "data": "", "texto": ""}
                        return (
                            msg,
                            "",
                            "",
                            msg,
                            "—",
                            "**Error** — ver consola del servidor",
                            empty_profile,
                        )

                doc_analyze_btn.click(
                    fn=run_doc_analysis,
                    inputs=[
                        doc_image,
                        doc_pdf,
                        doc_pdf_pages,
                        doc_pdf_dpi,
                        doc_clear_after,
                    ],
                    outputs=[
                        doc_graphic,
                        doc_texto_integral,
                        doc_data,
                        doc_raw,
                        doc_time,
                        doc_badge,
                        doc_profile_state,
                    ],
                    show_progress="full",
                )

            with gr.Tab("Documento dummy (POC)"):
                gr.Markdown(
                    "Plantilla **solo demostración**: bandas de color, degradado beige, patrón de círculos, "
                    "cabecera tipo cédula CO, datos a la izquierda y **foto a la derecha**, zona de **firma**. "
                    "El número se muestra con puntos (ej. `1.106.512.640`). "
                    "Si el análisis previo incluye un color en hex/RGB, la franja azul de título puede teñirse. "
                    "**No** reproduce hologramas ni elementos de seguridad reales."
                )
                with gr.Row():
                    with gr.Column():
                        dummy_nombre = gr.Textbox(
                            label="Nombres (ficticios)", value="Ricardo Alberto"
                        )
                        dummy_apellidos = gr.Textbox(
                            label="Apellidos (ficticios)", value="Richi Durango"
                        )
                        dummy_num = gr.Textbox(
                            label="N.º cédula (ficticio, solo dígitos o con puntos)",
                            value="1106512640",
                        )
                        dummy_fecha = gr.Textbox(
                            label="Fecha de nacimiento", value="15/08/1990"
                        )
                        dummy_nac = gr.Textbox(
                            label="Nacionalidad", value="Colombiana"
                        )
                        dummy_titulo = gr.Textbox(
                            label="Texto de cabecera (una línea larga o varias líneas)",
                            value=(
                                "REPÚBLICA DE COLOMBIA IDENTIFICACIÓN PERSONAL "
                                "CÉDULA DE CIUDADANIA"
                            ),
                        )
                        gen_dummy_btn = gr.Button(
                            "Generar documento dummy", variant="primary"
                        )
                    with gr.Column():
                        dummy_preview = gr.Image(label="Vista previa (sintética)")

                def do_dummy(
                    profile,
                    nombre,
                    apellidos,
                    num,
                    fecha,
                    nac,
                    titulo,
                ):
                    if not profile:
                        # Aún así generamos una tarjeta por defecto
                        profile = {"graphic": "", "data": "", "texto": ""}
                    im = render_dummy_id_card(
                        profile,
                        nombre,
                        apellidos,
                        num,
                        fecha,
                        nac,
                        titulo,
                    )
                    return im

                gen_dummy_btn.click(
                    fn=do_dummy,
                    inputs=[
                        doc_profile_state,
                        dummy_nombre,
                        dummy_apellidos,
                        dummy_num,
                        dummy_fecha,
                        dummy_nac,
                        dummy_titulo,
                    ],
                    outputs=[dummy_preview],
                )

    return demo


def main():
    resolve_device_and_attention()
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
