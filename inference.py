import argparse
import ctypes
import os
import queue
import threading
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from windows_capture import WindowsCapture


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "ffg_directml_fast.onnx" if Path("ffg_directml_fast.onnx").exists() else "ffgup.onnx"
)
STATIC_MODEL_PATH = os.environ.get("STATIC_MODEL_PATH", "ffgup_static.onnx")
DML_MODEL_PATH = os.environ.get("DML_MODEL_PATH", "ffg_directml_fast.onnx")
STATIC_DML_MODEL_PATH = os.environ.get("STATIC_DML_MODEL_PATH", "ffgup_static_directml.onnx")
INT8_MODEL_PATH = os.environ.get("INT8_MODEL_PATH", "ffg_directml_int8.onnx")
USE_INT8 = os.environ.get("USE_INT8", "0") == "1"
REBUILD_DML_MODEL = os.environ.get("REBUILD_DML_MODEL", "0") == "1"
MODEL_INPUT_SCALE = min(1.0, max(0.15, float(os.environ.get("MODEL_INPUT_SCALE", "0.15"))))
FAST_MOTION = True
MV_LEVELS = 1 if FAST_MOTION else 3
MV_WINSIZE = 7 if FAST_MOTION else 15
MV_ITERATIONS = 1 if FAST_MOTION else 3
USER32 = ctypes.windll.user32
GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040


def configure_overlay(title, target_title):
    overlay_hwnd = USER32.FindWindowW(None, title)
    target_hwnd = USER32.FindWindowW(None, target_title)
    if not overlay_hwnd or not target_hwnd:
        return None

    USER32.SetWindowLongW(overlay_hwnd, GWL_STYLE, WS_POPUP)
    current_exstyle = USER32.GetWindowLongW(overlay_hwnd, GWL_EXSTYLE)
    USER32.SetWindowLongW(
        overlay_hwnd,
        GWL_EXSTYLE,
        current_exstyle | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE,
    )
    return overlay_hwnd, target_hwnd


def move_overlay(overlay_hwnd, target_hwnd, previous_rect=None):
    rect = wintypes.RECT()
    if not USER32.GetClientRect(target_hwnd, ctypes.byref(rect)):
        return

    point = wintypes.POINT(rect.left, rect.top)
    if not USER32.ClientToScreen(target_hwnd, ctypes.byref(point)):
        return

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    current_rect = (point.x, point.y, width, height)
    if current_rect == previous_rect:
        return current_rect
    USER32.SetWindowPos(
        overlay_hwnd,
        HWND_TOPMOST,
        point.x,
        point.y,
        width,
        height,
        SWP_NOACTIVATE | SWP_SHOWWINDOW,
    )
    return current_rect


def find_window_title(partial_title):
    if not partial_title:
        return None

    found_title = None
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def enum_callback(hwnd, _lparam):
        nonlocal found_title
        if not USER32.IsWindowVisible(hwnd):
            return True
        length = USER32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        USER32.GetWindowTextW(hwnd, buffer, length + 1)
        if partial_title.casefold() in buffer.value.casefold():
            found_title = buffer.value
            return False
        return True

    USER32.EnumWindows(enum_callback, 0)
    return found_title


def prepare_directml_model(source_path, output_path):
    source = Path(source_path)
    output = Path(output_path)
    source_data = Path(f"{source_path}.data")
    output_data = Path(f"{output_path}.data")
    if output.exists() and output_data.exists() and not REBUILD_DML_MODEL:
        return str(output)

    source_mtime = max(path.stat().st_mtime for path in (source, source_data) if path.exists())

    if (
        output.exists()
        and output_data.exists()
        and output.stat().st_mtime >= source_mtime
        and output_data.stat().st_mtime >= source_mtime
    ):
        return str(output)

    model = onnx.load(str(source), load_external_data=True)
    for node in model.graph.node:
        if node.op_type == "Reshape":
            for attr in node.attribute:
                if attr.name == "allowzero" and attr.i == 1:
                    attr.i = 0

    output.unlink(missing_ok=True)
    output_data.unlink(missing_ok=True)
    onnx.save(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_data.name,
    )
    return str(output)


def prepare_int8_model(source_path, output_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = Path(source_path)
    output = Path(output_path)
    if output.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        return str(output)

    quantize_dynamic(
        str(source),
        str(output),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["Conv"],
    )
    return str(output)


def frame_to_input(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = frame_rgb.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) * 2.0
    return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None, ...])


def motion_vector_to_input(previous_gray, current, levels, winsize, iterations):
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow = np.clip(flow, -32, 32).astype(np.float32) / 16.0
    return np.ascontiguousarray(np.transpose(flow, (2, 0, 1))[None, ...]), current_gray


def output_to_frame(output, size):
    tensor = np.asarray(output[0], dtype=np.float32)[0]
    tensor = np.clip(tensor * 0.5 + 0.5, 0.0, 1.0)
    frame_rgb = np.transpose(tensor, (1, 2, 0))
    frame_bgr = cv2.cvtColor((frame_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)


def warmup_session(session, model_size, uses_motion_vectors):
    height, width = model_size[1], model_size[0]
    inputs = {
        "f0": np.zeros((1, 3, height, width), dtype=np.float32),
        "f1": np.zeros((1, 3, height, width), dtype=np.float32),
        "alpha": np.array([[[[0.5]]]], dtype=np.float32),
    }
    if uses_motion_vectors:
        inputs["mv"] = np.zeros((1, 2, height, width), dtype=np.float32)
    for _ in range(2):
        session.run(["output"], inputs)


def resolve_model_size(session, requested_size):
    shape = session.get_inputs()[0].shape
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int):
        return width, height
    return requested_size


def create_session(use_directml, use_int8, use_static=False):
    available = ort.get_available_providers()
    print(f"Available providers: {available}", flush=True)
    runtime_model = STATIC_MODEL_PATH if use_static and Path(STATIC_MODEL_PATH).exists() else MODEL_PATH
    providers = ["CPUExecutionProvider"]

    if use_directml and "DmlExecutionProvider" in available:
        source_model = runtime_model
        dml_model_path = (
            STATIC_DML_MODEL_PATH
            if use_static
            else DML_MODEL_PATH
        )
        runtime_model = prepare_directml_model(source_model, dml_model_path)
        if use_int8:
            runtime_model = prepare_int8_model(runtime_model, INT8_MODEL_PATH)
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    session_options.intra_op_num_threads = max(1, int(os.environ.get("ORT_INTRA_THREADS", "1")))
    session_options.inter_op_num_threads = 1

    try:
        session = ort.InferenceSession(
            runtime_model,
            sess_options=session_options,
            providers=providers,
        )
    except Exception as error:
        if not use_directml:
            raise
        if use_directml:
            print(f"Modelo INT8 no compatible con DirectML ({error}); probando FP32 en GPU", flush=True)
            runtime_model = prepare_directml_model(
                STATIC_MODEL_PATH if use_static and Path(STATIC_MODEL_PATH).exists() else MODEL_PATH,
                STATIC_DML_MODEL_PATH if use_static else DML_MODEL_PATH,
            )
            try:
                session = ort.InferenceSession(
                    runtime_model,
                    sess_options=session_options,
                    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
                )
            except Exception as dml_error:
                print(f"DirectML no pudo iniciar ({dml_error}); usando CPU", flush=True)
                runtime_model = STATIC_MODEL_PATH if use_static and Path(STATIC_MODEL_PATH).exists() else MODEL_PATH
                session = ort.InferenceSession(
                    runtime_model,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
        else:
            raise

    print(f"Using model: {runtime_model}", flush=True)
    print(f"Using providers: {session.get_providers()}", flush=True)
    return session


def main():
    parser = argparse.ArgumentParser(description="Inferencia FFG con Windows Graphics Capture")
    parser.add_argument("--window", default=os.environ.get("WGC_WINDOW_TITLE"), help="Titulo exacto de la ventana del juego")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor a capturar (empieza en 1) si no se indica --window")
    parser.add_argument("--cpu", action="store_true", help="Desactiva DirectML")
    parser.add_argument("--int8", action="store_true", help="Usa el modelo cuantizado INT8; puede ser mas lento en DirectML")
    parser.add_argument("--static", action="store_true", help="Usa ONNX estatico 480x270 exportado por ffgup.py")
    parser.add_argument("--alpha", type=float, default=0.5, help="Posicion del frame interpolado, 0..1")
    parser.add_argument(
        "--scale",
        type=float,
        default=MODEL_INPUT_SCALE,
        help="Escala de entrada; 0.15 es la opcion mas fluida, 0.25 es equilibrada, 0.5 tiene mas detalle pero mas lag",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Objetivo de procesamiento en FPS; por defecto 30 para mantener la latencia por debajo del frame budget")
    parser.add_argument("--full-motion", action="store_true", help="Usa parametros de motion vectors de mayor calidad; aumenta el coste")
    parser.add_argument("--stats", action="store_true", help="Muestra tiempos medios de resize, motion vectors, inferencia y salida")
    parser.add_argument("--no-overlay", action="store_true", help="Muestra una ventana normal en vez de superponerla al juego")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha debe estar entre 0 y 1")
    if not 0.15 <= args.scale <= 1.0:
        parser.error("--scale debe estar entre 0.15 y 1")
    if not 1.0 <= args.fps <= 120.0:
        parser.error("--fps debe estar entre 1 y 120")

    if args.window:
        resolved_title = find_window_title(args.window)
        if resolved_title is None:
            raise RuntimeError(f"No se encontro una ventana que contenga: {args.window!r}")
        if resolved_title != args.window:
            print(f"Ventana encontrada: {resolved_title!r}", flush=True)
            args.window = resolved_title

    use_int8 = USE_INT8 or args.int8
    if use_int8:
        print("INT8 activado: DirectML puede ejecutar algunas capas cuantizadas mas lentamente", flush=True)
    use_static = args.static
    if use_static and not Path(STATIC_MODEL_PATH).exists():
        raise RuntimeError(
            f"No existe {STATIC_MODEL_PATH}. Ejecuta ffgup.py para exportarlo."
        )
    session = create_session(not args.cpu, use_int8, use_static)
    input_names = {item.name for item in session.get_inputs()}
    required = {"f0", "f1", "alpha"}
    if not required.issubset(input_names):
        raise RuntimeError(f"Entradas ONNX inesperadas: {sorted(input_names)}")
    uses_motion_vectors = "mv" in input_names
    if uses_motion_vectors:
        print("Modo juego: motion vectors activos", flush=True)
    else:
        print("Modo juego: motion vectors desactivados para reducir la latencia", flush=True)

    frames = queue.Queue(maxsize=1)
    closed = threading.Event()

    capture = WindowsCapture(
        cursor_capture=False,
        draw_border=False,
        minimum_update_interval=16,
        monitor_index=None if args.window else args.monitor,
        window_name=args.window,
    )

    @capture.event
    def on_frame_arrived(frame, capture_control):
        image = np.ascontiguousarray(frame.convert_to_bgr().frame_buffer).copy()
        try:
            frames.put_nowait(image)
        except queue.Full:
            try:
                frames.get_nowait()
            except queue.Empty:
                pass
            try:
                frames.put_nowait(image)
            except queue.Full:
                pass

    @capture.event
    def on_closed():
        closed.set()

    capture_control = capture.start_free_threaded()
    overlay_title = "FFG WGC overlay"
    overlay_target = None
    if args.window and not args.no_overlay:
        cv2.namedWindow(overlay_title, cv2.WINDOW_NORMAL)
        overlay_target = configure_overlay(overlay_title, args.window)
        if overlay_target is None:
            cv2.destroyWindow(overlay_title)
            raise RuntimeError(f"No se encontro la ventana del juego: {args.window!r}")

    previous = None
    previous_model = None
    previous_input = None
    previous_gray = None
    frame_size = None
    model_size = None
    alpha = np.array([[[[args.alpha]]]], dtype=np.float32)
    frame_interval = 1.0 / args.fps
    last_process_time = 0.0
    overlay_rect = None
    overlay_check_count = 0
    stats_count = 0
    stats_totals = {"resize": 0.0, "motion": 0.0, "inference": 0.0, "output": 0.0}

    print(
        f"Capturando {'ventana ' + repr(args.window) if args.window else 'monitor ' + str(args.monitor)}. "
        f"Objetivo: {args.fps:.0f} FPS. Pulsa Ctrl+C para salir.",
        flush=True,
    )

    try:
        while not closed.is_set():
            try:
                current = frames.get(timeout=0.25)
            except queue.Empty:
                continue

            now = time.perf_counter()
            if last_process_time > 0 and (now - last_process_time) < frame_interval:
                continue

            # No proceses frames viejos: la latencia importa mas que conservar cada frame.
            while True:
                try:
                    current = frames.get_nowait()
                except queue.Empty:
                    break

            if previous is None:
                previous = current
                height, width = current.shape[:2]
                frame_size = (width - width % 2, height - height % 2)
                requested_model_size = (
                    max(16, int(frame_size[0] * args.scale) // 2 * 2),
                    max(16, int(frame_size[1] * args.scale) // 2 * 2),
                )
                model_size = resolve_model_size(session, requested_model_size)
                previous_model = cv2.resize(previous, model_size, interpolation=cv2.INTER_AREA)
                previous_input = frame_to_input(previous_model)
                previous_gray = cv2.cvtColor(previous_model, cv2.COLOR_BGR2GRAY)
                print(
                    f"Captura: {frame_size[0]}x{frame_size[1]}, modelo: "
                    f"{model_size[0]}x{model_size[1]}",
                    flush=True,
                )
                warmup_session(session, model_size, uses_motion_vectors)
                print("DirectML calentado", flush=True)
                continue

            resize_started = time.perf_counter()
            current_model = cv2.resize(current, model_size, interpolation=cv2.INTER_AREA)
            stats_totals["resize"] += time.perf_counter() - resize_started
            current_input = frame_to_input(current_model)
            inputs = {
                "f0": previous_input,
                "f1": current_input,
                "alpha": alpha,
            }
            if uses_motion_vectors:
                motion_started = time.perf_counter()
                if args.full_motion:
                    motion_levels, motion_winsize, motion_iterations = 2, 11, 2
                else:
                    motion_levels, motion_winsize, motion_iterations = MV_LEVELS, MV_WINSIZE, MV_ITERATIONS
                inputs["mv"], current_gray = motion_vector_to_input(
                    previous_gray,
                    current_model,
                    motion_levels,
                    motion_winsize,
                    motion_iterations,
                )
                stats_totals["motion"] += time.perf_counter() - motion_started
            elif "mv" in input_names:
                inputs["mv"] = np.zeros((1, 2, model_size[1], model_size[0]), dtype=np.float32)

            inference_started = time.perf_counter()
            output = session.run(["output"], inputs)
            inference_elapsed = time.perf_counter() - inference_started
            last_process_time = time.perf_counter()
            stats_totals["inference"] += inference_elapsed

            output_started = time.perf_counter()
            result = output_to_frame(output, frame_size)
            stats_totals["output"] += time.perf_counter() - output_started
            if args.window and not args.no_overlay:
                overlay_check_count += 1
                if overlay_check_count >= 15:
                    overlay_rect = move_overlay(*overlay_target, overlay_rect)
                    overlay_check_count = 0
                cv2.imshow(overlay_title, result)
            # En modo normal no se abre una segunda ventana con la imagen.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) and args.no_overlay:
                break

            previous = current
            previous_model = current_model
            previous_input = current_input
            previous_gray = current_gray if uses_motion_vectors else None
            stats_count += 1
            if args.stats and stats_count >= 30:
                print(
                    "Stats ms/frame: "
                    + ", ".join(
                        f"{name}={stats_totals[name] * 1000.0 / stats_count:.2f}"
                        for name in ("resize", "motion", "inference", "output")
                    ),
                    flush=True,
                )
                stats_count = 0
                stats_totals = {"resize": 0.0, "motion": 0.0, "inference": 0.0, "output": 0.0}
    finally:
        capture_control.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
