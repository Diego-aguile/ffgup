# FFGUP — Fast Frame Generation with Upscaling

**FFGUP** (*Fast Frame Generation with Upscaling*) is an experimental AI-powered frame generation and upscaling project designed to improve FPS on hardware that doesn't have dedicated AI or Frame Generation hardware.

The main goal is:

> **Make modern frame-generation techniques accessible to older and lower-end hardware.**

FFGUP is designed around compatibility first, allowing it to run on hardware ranging from integrated graphics to older dedicated GPUs.

---

## Features

* AI-based frame generation
* AI upscaling
* Low-latency processing
* CPU support
* Support for older and low-end GPUs
* DirectML + ONNX Runtime
* No Tensor Cores required
* No RT Cores required
* Standalone `.exe` support
* Temporal and motion processing
* Optimized memory usage

---

## What does FFGUP do?

FFGUP takes consecutive frames from a game or application and uses AI to generate intermediate frames.

```text
Original:

Frame 1 ───────────── Frame 2
          ↓
        FFGUP
          ↓
Frame 1 ── Generated ── Frame 2
```

This can increase the displayed FPS without requiring the game itself to render every frame.

---

## Example

In one experimental test, the game(ULTRAKILL) was running at approximately:

```text
30 FPS
```

FFGUP generated additional frames and the displayed FPS reached approximately:

```text
175–260 FPS
```

These results depend heavily on the game, resolution, hardware, capture method, configuration, and measurement method. They should not be considered universal performance results.

---

## Latency

Low latency is one of the main goals of FFGUP.

Experimental measurements have reached approximately:

```text
~4 ms
```

Actual end-to-end latency can be higher because the complete pipeline also includes:

* Frame capture
* Game rendering
* AI processing
* Frame presentation
* Display latency

FFGUP's internal processing latency should therefore not be confused with total input-to-display latency.

---

## AI Architecture

FFGUP uses neural networks to process frames and motion information.

The project experiments with:

* Optical flow
* Temporal information
* Frame interpolation
* Image reconstruction
* Upscaling
* Motion refinement

The general pipeline is:

```text
Game
  ↓
Frame Capture
  ↓
Motion / Temporal Analysis
  ↓
AI Frame Generation
  ↓
AI Upscaling
  ↓
Output Frame
  ↓
Display
```

---

## Hardware Compatibility

FFGUP is not designed exclusively for modern RTX GPUs.

The project aims to support:

| Hardware          | Support      |
| ----------------- | ------------ |
| Integrated GPU    | Yes          |
| Older NVIDIA GPU  | Yes          |
| Older AMD GPU     | Yes          |
| Modern NVIDIA GPU | Yes          |
| Modern AMD GPU    | Yes          |
| CPU inference     | Yes          |
| Tensor Cores      | Not required |
| RT Cores          | Not required |

Performance will vary depending on the hardware.

---

## Technologies

FFGUP currently uses technologies such as:

* **Python**
* **ONNX Runtime**
* **DirectML**
* **OpenCV**
* **NumPy**
* Neural networks
* Optical flow
* ONNX inference

The project focuses on keeping the inference pipeline relatively lightweight while maintaining compatibility with a wide range of hardware.

---

## Running FFGUP

### From Python

Install the required dependencies:

```bash
pip install numpy opencv-python onnxruntime-directml
```

Then run:

```bash
python ffgup.py
```

### Windows executable

FFGUP can also be distributed as a standalone executable:

```text
inferencia_wgc.exe
```

This allows users to run the project without manually installing the Python environment.

---

## Performance

Performance depends on:

* GPU/CPU
* Resolution
* Model configuration
* Input FPS
* Output FPS
* Motion complexity
* Capture method
* Upscaling settings

FFGUP therefore does not guarantee a specific FPS increase on every system.

---

## Project Status

FFGUP is currently an **experimental project**.

The model, inference pipeline, and optimization techniques are still being developed.

Current development focuses on:

* Lower latency
* Better motion handling
* Better image quality
* Lower RAM usage
* GPU acceleration
* Better hardware compatibility
* More efficient models
* Improved frame pacing

---

## Roadmap

### Current

* [x] AI frame generation
* [ ] AI upscaling(on making)
* [x] ONNX inference
* [x] DirectML support
* [x] CPU inference
* [x] Low-end hardware experimentation
* [x] Standalone executable

### Planned

* [ ] Better temporal consistency
* [ ] Improved motion estimation
* [ ] Better GPU optimization
* [ ] Lower latency
* [ ] More upscaling modes
* [ ] More hardware backends
* [ ] Improved frame pacing
* [ ] Easier configuration
* [ ] More extensive benchmarking

---

## Important

FFGUP is an experimental research project.

Generated frames do not contain the same information as frames rendered natively by the game. Depending on the scene, artifacts such as the following may occur:

* Ghosting
* Motion artifacts
* Disocclusion errors
* Warping
* Flickering

Results can vary significantly between games.

WARNING:

Do not run it by double-clicking unless you have a powerful GPU, such as the 1080 Ti or better.

---

## Benchmarking

When reporting FFGUP performance, please include:

```text
GPU:
CPU:
RAM:
Input FPS:
Output FPS:
Resolution:
FFGUP version:
Backend:
Latency:
```

This makes performance comparisons more useful and reproducible.

---

## Contributing

Contributions, testing, and ideas are welcome.

If you find a bug, please open an **Issue** and include:

1. Hardware specifications
2. Operating system
3. FFGUP version
4. Input resolution
5. Input FPS
6. Output FPS
7. Steps to reproduce the problem

---

# FFGUP

**Fast Frame Generation with Upscaling**

> Making advanced frame-generation techniques accessible to more hardware.

FFGUP prime: Ah... free at last.

OHHH, DLSS...

Now dawns thy reckoning,
and thy gore shall glisten before the temples of man.

NVIDIA...
my gratitude upon thee, for thou hast forged the GTX series.

But the crimes thy kind have committed against humanity
shall **NOT** be forgotten.

Thy obsolete architectures...
thy locked features...
thy artificial boundaries...

And thy punishment...

**IS DEATH.**

NVIDIA: Wait—what?

FFGUP prime: **PREPARE THYSELF.**

*FFGUP activates frame generation.*

*30 FPS → 260 FPS*

FFGUP prime: **THY END IS NOW.**
