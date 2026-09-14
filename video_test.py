import cv2
import onnxruntime as ort
import numpy as np
import onnx
import os
import time
from pathlib import Path
import random

# =========================================
# SETTINGS
# =========================================
MODEL_PATH = "ffgup.onnx"
DML_MODEL_PATH = "ffg_directml_fast.onnx"
INPUT_VIDEO = random.choice(["fn.mp4", "MC.mp4", "ow 2.mp4", "indie.mp4"])
OUTPUT_VIDEO = "output1.mp4"
INTERPOLATION_ALPHA = 0.5
LOW_RAM_MODE = True
MODEL_INPUT_SCALE = min(
    1.0,
    max(0.5, float(os.environ.get("MODEL_INPUT_SCALE", "0.625")))
)
USE_DIRECTML = os.environ.get("USE_DIRECTML", "0") == "1"
FAST_MOTION = True
MV_LEVELS = 2 if FAST_MOTION else 3
MV_WINSIZE = 11 if FAST_MOTION else 15
MV_ITERATIONS = 2 if FAST_MOTION else 3
MAX_FRAME_PAIRS = int(os.environ.get("MAX_FRAME_PAIRS", "0"))
ORT_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "0"))
ORT_INTER_THREADS = int(os.environ.get("ORT_INTER_THREADS", "1"))

# =========================================
# MODEL
# =========================================
available_providers = ort.get_available_providers()
print("Available providers:", available_providers, flush=True)

if not LOW_RAM_MODE and "DmlExecutionProvider" not in available_providers:
    raise RuntimeError(
        "DmlExecutionProvider no esta disponible. "
        "Instala onnxruntime-directml para usar DirectML."
    )


def prepare_directml_model(source_path, output_path):
    source = Path(source_path)
    output = Path(output_path)
    source_data = Path(f"{source_path}.data")
    output_data = Path(f"{output_path}.data")

    source_mtime = max(
        p.stat().st_mtime
        for p in (source, source_data)
        if p.exists()
    )

    if (
        output.exists()
        and output_data.exists()
        and output.stat().st_mtime >= source_mtime
        and output_data.stat().st_mtime >= source_mtime
    ):
        return str(output)

    model = onnx.load(str(source), load_external_data=True)
    changed = 0

    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue

        for attr in node.attribute:
            if attr.name == "allowzero" and attr.i == 1:
                attr.i = 0
                changed += 1

    if changed:
        print(
            f"DirectML ONNX fix: Reshape allowzero ajustado en {changed} nodos",
            flush=True
        )

    try:
        output.unlink(missing_ok=True)
        output_data.unlink(missing_ok=True)
    except PermissionError:
        if output.exists():
            print(f"{output} bloqueado; usando el modelo DirectML existente", flush=True)
            return str(output)
        raise

    onnx.save(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_data.name
    )

    return str(output)


use_directml = USE_DIRECTML and "DmlExecutionProvider" in available_providers
runtime_model_path = (
    prepare_directml_model(MODEL_PATH, DML_MODEL_PATH)
    if use_directml
    else MODEL_PATH
)

def frame_to_onnx_input(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = frame_rgb.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) * 2.0
    return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None, ...])


def motion_vector_to_onnx_input(prev_bgr, curr_bgr):
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=MV_LEVELS,
        winsize=MV_WINSIZE,
        iterations=MV_ITERATIONS,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    flow = np.clip(flow, -32, 32).astype(np.float32) / 16.0
    return np.ascontiguousarray(np.transpose(flow, (2, 0, 1))[None, ...])


def resize_for_model(frame, size):
    if (frame.shape[1], frame.shape[0]) == size:
        return frame

    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def onnx_output_to_frame(output, size):
    tensor = np.asarray(output[0], dtype=np.float32)
    tensor = (tensor * 0.5 + 0.5)
    frame_rgb = np.transpose(tensor, (1, 2, 0))
    frame_rgb = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if (frame_bgr.shape[1], frame_bgr.shape[0]) != size:
        frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)

    return prepare_writer_frame(frame_bgr, size)


def prepare_writer_frame(frame, size):
    if frame is None:
        raise RuntimeError("Frame invalido para escribir")

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(frame)


def main():
    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = not LOW_RAM_MODE
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = ORT_INTRA_THREADS
    session_options.inter_op_num_threads = ORT_INTER_THREADS

    providers = (
        ["DmlExecutionProvider", "CPUExecutionProvider"]
        if use_directml
        else ["CPUExecutionProvider"]
    )

    try:
        session = ort.InferenceSession(
            runtime_model_path,
            sess_options=session_options,
            providers=providers
        )
    except Exception:
        if not use_directml:
            raise
        print("DirectML no pudo iniciar; usando CPU", flush=True)
        session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )
    print("Using providers:", session.get_providers(), flush=True)
    print("Model:", runtime_model_path, flush=True)

    input_names = {inp.name for inp in session.get_inputs()}
    if not {"f0", "f1", "alpha"}.issubset(input_names):
        raise RuntimeError(f"Entradas ONNX inesperadas: {sorted(input_names)}")
    uses_motion_vectors = "mv" in input_names

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev = cap.read()

    if not ret:
        cap.release()
        raise RuntimeError("Failed to open video")

    h, w = prev.shape[:2]
    writer_size = (
        w - (w % 2),
        h - (h % 2)
    )
    model_size = (
        max(16, int(writer_size[0] * MODEL_INPUT_SCALE) // 2 * 2),
        max(16, int(writer_size[1] * MODEL_INPUT_SCALE) // 2 * 2)
    )
    print(w, h)

    print(
        f"Interpolando {INPUT_VIDEO}: {w}x{h}, {fps:.2f} FPS -> {fps * 2:.2f} FPS",
        flush=True
    )
    print(f"Modelo: {model_size[0]}x{model_size[1]} LOW_RAM_MODE={LOW_RAM_MODE}", flush=True)

    output_video = OUTPUT_VIDEO
    try:
        Path(output_video).unlink(missing_ok=True)
    except PermissionError:
        output_path = Path(OUTPUT_VIDEO)
        output_video = str(output_path.with_name(f"{output_path.stem}_lowram{output_path.suffix}"))
        Path(output_video).unlink(missing_ok=True)
        print(f"{OUTPUT_VIDEO} bloqueado; usando {output_video}", flush=True)

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps * 2,
        writer_size
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear {output_video}")

    frame_id = 0
    latency_total_ms = 0.0
    alpha_np = np.array([[[[INTERPOLATION_ALPHA]]]], dtype=np.float32)

    while True:
        ret, curr = cap.read()

        if not ret:
            break

        writer.write(prepare_writer_frame(prev, writer_size))

        prev_model = resize_for_model(prev, model_size)
        curr_model = resize_for_model(curr, model_size)

        f0_np = frame_to_onnx_input(prev_model)
        f1_np = frame_to_onnx_input(curr_model)
        inputs = {
            "f0": f0_np,
            "f1": f1_np,
            "alpha": alpha_np
        }

        if uses_motion_vectors:
            inputs["mv"] = motion_vector_to_onnx_input(prev_model, curr_model)

        start = time.perf_counter()
        out = session.run(
            ["output"],
            inputs
        )[0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        latency_total_ms += latency_ms

        writer.write(onnx_output_to_frame(out, writer_size))

        del prev_model, curr_model, f0_np, f1_np, inputs, out
        prev = curr
        frame_id += 1

        if frame_id % 30 == 0:
            avg_latency_ms = latency_total_ms / max(frame_id, 1)
            if total_frames > 0:
                print(
                    f"Processed {frame_id}/{total_frames - 1} frame pairs "
                    f"latencia media {avg_latency_ms:.2f} ms",
                    flush=True
                )
            else:
                print(
                    f"Processed {frame_id} frame pairs "
                    f"latencia media {avg_latency_ms:.2f} ms",
                    flush=True
                )

        if MAX_FRAME_PAIRS and frame_id >= MAX_FRAME_PAIRS:
            avg_latency_ms = latency_total_ms / max(frame_id, 1)
            print(
                f"Corte de prueba en {frame_id} pares; "
                f"latencia media {avg_latency_ms:.2f} ms",
                flush=True
            )
            break

    writer.write(prepare_writer_frame(prev, writer_size))

    cap.release()
    writer.release()

    print(f"Done! Guardado en {output_video}", flush=True)


if __name__ == "__main__":
    main()
