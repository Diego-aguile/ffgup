import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os
import queue
import threading
import time
from collections import OrderedDict
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)
from onnxruntime.quantization import quantize_dynamic, QuantType

# =========================================
# DEVICE
# =========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
HAS_GPU = device == "cuda"

AGGRESSIVE_PROFILE = os.environ.get("AGGRESSIVE_PROFILE", "0") == "1"
LOW_RAM_MODE = os.environ.get("LOW_RAM_MODE", "1") == "1"
FRAME_SIZE = (960, 600) if LOW_RAM_MODE else (1280, 800)
SAMPLES_PER_CLIP = 4 if LOW_RAM_MODE else 10
FAST_MOTION = True
MV_LEVELS = 2 if FAST_MOTION else 3
MV_WINSIZE = 11 if FAST_MOTION else 15
MV_ITERATIONS = 2 if FAST_MOTION else 3
REFLECTION_AUGMENTATION = os.environ.get("REFLECTION_AUGMENTATION", "1") == "1"
REFLECTION_PROBABILITY = min(
    1.0,
    max(0.0, float(os.environ.get("REFLECTION_PROBABILITY", "0.75")))
)
TEMPORAL_CONSISTENCY_WEIGHT = float(
    os.environ.get("TEMPORAL_CONSISTENCY_WEIGHT", "0.06")
)
TEMPORAL_CHECK_INTERVAL = max(
    1,
    int(os.environ.get("TEMPORAL_CHECK_INTERVAL", "4"))
)
ANTI_GHOSTING = os.environ.get("ANTI_GHOSTING", "1") == "1"
ANTI_GHOST_THRESHOLD = float(os.environ.get("ANTI_GHOST_THRESHOLD", "0.035"))
ANTI_GHOST_GAIN = float(os.environ.get("ANTI_GHOST_GAIN", "10.0"))
OCCLUSION_FLOW_WEIGHT = float(os.environ.get("OCCLUSION_FLOW_WEIGHT", "0.10"))
OCCLUSION_CONSISTENCY_WEIGHT = float(
    os.environ.get("OCCLUSION_CONSISTENCY_WEIGHT", "0.015")
)
PREFETCH_BATCHES = max(
    1,
    int(os.environ.get("PREFETCH_BATCHES", "2"))
)
INFERENCE_BATCH_SIZE = max(
    1,
    int(os.environ.get("INFERENCE_BATCH_SIZE", "4"))
)

use_fp16 = device == "cuda"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "0"))
if TORCH_NUM_THREADS > 0:
    torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(1)

#==========================================
# DATASHEET
#==========================================
VIDEO = "fn.mp4"
VIDEO2 = "MC.mp4"
VIDEO3 = "ow 2.mp4"
VIDEO4 = "indie.mp4"
DATASET_DIR = "dataset"
VIDEO_A_DIR = os.path.join(DATASET_DIR, "videoA")
VIDEO_B_DIR = os.path.join(DATASET_DIR, "videoB")
VIDEO_C_DIR = os.path.join(DATASET_DIR, "video3")
VIDEO_D_DIR = os.path.join(DATASET_DIR, "video4")
os.makedirs(VIDEO_A_DIR, exist_ok=True)
os.makedirs(VIDEO_B_DIR, exist_ok=True)
os.makedirs(VIDEO_C_DIR, exist_ok=True)
os.makedirs(VIDEO_D_DIR, exist_ok=True)
def extract(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        cv2.imwrite(
            os.path.join(out_dir, f"{i:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        i += 1

    cap.release()
    print(f"{video_path}: {i} frames")
if len(os.listdir(VIDEO_A_DIR)) == 0:
    extract(VIDEO, VIDEO_A_DIR)


if len(os.listdir(VIDEO_B_DIR)) == 0:
    extract(VIDEO2, VIDEO_B_DIR)

if len(os.listdir(VIDEO_C_DIR)) == 0:
    extract(VIDEO3, VIDEO_C_DIR)

if len(os.listdir(VIDEO_D_DIR)) == 0:
    extract(VIDEO4, VIDEO_D_DIR)

def load_video_frames(path):
    return sorted([
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

videoA = load_video_frames(VIDEO_A_DIR)
videoB = load_video_frames(VIDEO_B_DIR)
videoC = load_video_frames(VIDEO_C_DIR)
videoD = load_video_frames(VIDEO_D_DIR)

print("Video A frames:", len(videoA))
print("Video B frames:", len(videoB))
print("Video C frames:", len(videoC))
print("Video D frames:", len(videoD))
# =========================================
# MOTION VECTORS
# =========================================
FRAME_CACHE_SIZE = max(
    0,
    int(os.environ.get("FRAME_CACHE_SIZE", "32" if LOW_RAM_MODE else "128"))
)
MV_CACHE_SIZE = max(
    0,
    int(os.environ.get("MV_CACHE_SIZE", "8" if LOW_RAM_MODE else "32"))
)
frame_cache = OrderedDict()
mv_cache = OrderedDict()

def load_cached_frame(path):
    if FRAME_CACHE_SIZE == 0:
        return cv2.imread(path)

    if path in frame_cache:
        frame_cache.move_to_end(path)
        return frame_cache[path]

    frame = cv2.imread(path)
    if frame is None:
        return None

    frame_cache[path] = frame
    frame_cache.move_to_end(path)
    while len(frame_cache) > FRAME_CACHE_SIZE:
        frame_cache.popitem(last=False)
    return frame

def compute_mv(f0, f1):
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        g0, g1,
        None,
        pyr_scale=0.5,
        levels=MV_LEVELS,
        winsize=MV_WINSIZE,
        iterations=MV_ITERATIONS,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    return flow
def get_mv(frame0, frame1, key):
    if MV_CACHE_SIZE == 0:
        return compute_mv(frame0, frame1)

    if key in mv_cache:
        mv_cache.move_to_end(key)
        return mv_cache[key]

    flow = compute_mv(frame0, frame1)

    mv_cache[key] = flow
    mv_cache.move_to_end(key)
    while len(mv_cache) > MV_CACHE_SIZE:
        mv_cache.popitem(last=False)

    return flow

def save_mv_dataset(frame_paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(frame_paths) - 1):
        f0 = cv2.imread(frame_paths[i])
        f1 = cv2.imread(frame_paths[i + 1])

        if f0 is None or f1 is None:
            continue

        flow = compute_mv(f0, f1)

        flow = np.clip(flow, -32, 32)
        flow = ((flow + 32) / 64 * 65535).astype(np.uint16)
        
        dummy = np.zeros((flow.shape[0], flow.shape[1], 1), dtype=np.uint16)
        flow_png = np.concatenate([flow, dummy], axis=2)
        
        cv2.imwrite(
            os.path.join(out_dir, f"{i:06d}.png"),
            flow_png
        )

        print("saved mv", i)
# =========================================
# DATA
# =========================================
def generate_data(batch=16):
    videos = [
    (videoA, "mv/videoA"),
    (videoB, "mv/videoB"),
    (videoC, "mv/videoC"),
    (videoD, "mv/videoD")
    ]

    for _ in range(batch):

        frames, mv_dir = random.choice(videos)
        if len(frames) < 9:
            continue
        idx = np.random.randint(0, len(frames) - 8)

        f0 = load_cached_frame(frames[idx])
        f1 = load_cached_frame(frames[idx + 8])
        offset = np.random.randint(1, 8)
        ft = load_cached_frame(frames[idx + offset])

        alpha = offset / 8.0

        if f0 is None or ft is None or f1 is None:
            continue

        key = (mv_dir, idx)

        mv_full = get_mv(
            f0,
            f1,
            key
        )

        H, W = f0.shape[:2]
        patch = 100

        for _ in range(SAMPLES_PER_CLIP):
            x = np.random.randint(0, W - patch)
            y = np.random.randint(0, H - patch)

            p0 = f0[y:y+patch, x:x+patch]
            pt = ft[y:y+patch, x:x+patch]
            p1 = f1[y:y+patch, x:x+patch]
            mv = mv_full[y:y+patch, x:x+patch]

            if REFLECTION_AUGMENTATION and random.random() < REFLECTION_PROBABILITY:
                p0 = np.ascontiguousarray(np.flip(p0, axis=1))
                pt = np.ascontiguousarray(np.flip(pt, axis=1))
                p1 = np.ascontiguousarray(np.flip(p1, axis=1))
                mv = np.ascontiguousarray(np.flip(mv, axis=1).copy())
                mv[:, :, 0] *= -1.0

            yield p0, p1, pt, alpha, mv


class AsyncBatchPrefetcher:

    def __init__(self, sample_factory, batch_size, queue_size):
        self.sample_factory = sample_factory
        self.batch_size = batch_size
        self.samples = queue.Queue(maxsize=queue_size)
        self.sentinel = object()

    def _produce(self):
        try:
            batch = []
            for sample in self.sample_factory():
                batch.append(sample)
                if len(batch) == self.batch_size:
                    self.samples.put(batch)
                    batch = []
            if batch:
                self.samples.put(batch)
        except BaseException as error:
            self.samples.put(error)
        finally:
            self.samples.put(self.sentinel)

    def __iter__(self):
        producer = threading.Thread(
            target=self._produce,
            name="ffgup-data-prefetch",
            daemon=True,
        )
        producer.start()
        while True:
            item = self.samples.get()
            if item is self.sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

# =========================================
# UTILS
# =========================================
def np_to_tensor(img):

    t = torch.from_numpy(img)\
        .permute(2,0,1)\
        .float() / 255.0

    t = (t - 0.5) * 2.0

    return t


def tensor_to_np(t):

    t = (t * 0.5 + 0.5)

    img = t.permute(1,2,0)\
        .detach()\
        .float()\
        .cpu()\
        .numpy()

    return np.clip(
        img * 255,
        0,
        255
    ).astype(np.uint8)


def diff_image(pred, gt):

    diff = cv2.absdiff(pred, gt)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    return cv2.applyColorMap(
        diff_gray,
        cv2.COLORMAP_JET
    )

# =========================================
# WARP + GRID CACHE
# =========================================

_grid_cache = {}

def get_grid(H, W, device):

    if torch.onnx.is_in_onnx_export():
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        return torch.stack((x, y), dim=-1).float()

    key = (H, W, str(device))

    if key not in _grid_cache:
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        _grid_cache[key] = torch.stack((x, y), dim=-1).float()

    return _grid_cache[key]

def warp(img, flow):

    B, C, H, W = img.shape

    grid = get_grid(H, W, img.device)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    flow = flow.permute(0,2,3,1)

    new_grid = grid + flow

    new_grid[...,0] = (
        2.0 * new_grid[...,0] / (W-1)
    ) - 1.0

    new_grid[...,1] = (
        2.0 * new_grid[...,1] / (H-1)
    ) - 1.0
    
    return F.grid_sample(
        img,
        new_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )

# =========================================
# MODEL + DIFICULTY
# =========================================
class FlowNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                12,
                96,
                3, 
                padding=1
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                96,
                96,
                3,
                stride=2,
                padding=1
                ),
            
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                96,
                96,
                3,
                stride=2,
                padding=1
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                96,
                96,
                3,
                padding=1
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                96,
                5,
                3,
                padding=1
                ),
        )
        self.refine = nn.Sequential(

            nn.Conv2d(
                25,
                80, 
                3, 
                padding=1
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                80,
                80, 
                3, 
                padding=2,
                dilation=2,
                groups=80
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                80,
                80, 
                3,
                padding=1
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                80,
                80, 
                3,
                padding=2,
                dilation=2,
                groups=80
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                80,
                80, 
                3,
                padding=1,
                groups=80
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                80,
                80, 
                3, 
                padding=1,
                groups=80
                ),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(
                80,
                3, 
                3, 
                padding=1
                )
        )

    def forward(self, f0, f1, mv,  alpha):

        B, _, H, W = f0.shape

        a = torch.ones(
            (B,1,H,W),
            device=f0.device,
            dtype=f0.dtype
        ) * alpha

        diff = f1 - f0

        x = torch.cat([f0, f1, diff, mv, a], dim=1)

        out = self.net(x)

        delta_scale = 6.0

        delta0 = torch.tanh(out[:,0:2]) * delta_scale
        delta1 = torch.tanh(out[:,2:4]) * delta_scale
        
        mask = torch.sigmoid(out[:,4:5])
        
        delta0 = F.interpolate(
            delta0,
            size=(H,W),
            mode="bilinear"
            
        )
        
        delta1 = F.interpolate(
            delta1,
            size=(H,W),
            mode="bilinear"
            
        )
        
        mask = F.interpolate(
            mask,
            size=(H,W),
            mode="bilinear"
            
        )
        
        flow0 = mv * alpha + delta0
        flow1 = -mv * (1.0 - alpha) + delta1

        out0 = warp(f0, flow0)
        out1 = warp(f1, flow1)

        disagreement = torch.mean(torch.abs(out0 - out1), dim=1, keepdim=True)
        if ANTI_GHOSTING:
            ghost_gate = torch.clamp(
                (disagreement - ANTI_GHOST_THRESHOLD) * ANTI_GHOST_GAIN,
                0.0,
                1.0
            )
            hard_mask = (mask >= 0.5).to(mask.dtype)
            mask2 = mask * (1.0 - ghost_gate) + hard_mask.detach() * ghost_gate
        else:
            agreement = torch.exp(-8.0 * disagreement)
            mask2 = mask * agreement
            mask2 = mask2 / (mask2 + (1.0 - mask) * agreement + 1e-6)

        blend = mask2 * out0 + (1.0 - mask2) * out1
        blend_up = F.interpolate(
            blend,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        motion = torch.abs(out0 - out1)
        refine_input = torch.cat([
            flow0,
            flow1,
            mv,
            mask,
            motion,
            out0,
            out1,
            blend,
            f0,
            f1
        ], dim=1)

        refine_size = (
            blend.shape[-2] // 2,
            blend.shape[-1] // 2
        )
        refine_input = F.interpolate(
            refine_input,
            size=refine_size,
            mode="bilinear",
            align_corners=False
        )

        residual = self.refine(refine_input)

        residual = F.interpolate(
            residual,
            size=blend.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        residual = torch.tanh(residual) * 0.2
        
        refined = blend + residual
        refined = F.interpolate(
            refined,
            size=blend_up.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        final = torch.clamp(
            refined,
            -1,
            1
        )

        return final, out0, out1
    


# =========================================
# LOSS
# =========================================
def ssim_loss(pred, gt):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(gt, 3, 1, 1)

    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(gt * gt, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * gt, 3, 1, 1) - mu_x * mu_y

    ssim = (
        (2 * mu_x * mu_y + C1)
        * (2 * sigma_xy + C2)
    ) / (
        (mu_x * mu_x + mu_y * mu_y + C1)
        * (sigma_x + sigma_y + C2)
    )

    return 1.0 - ssim.mean()

def occlusion_aware_warp_losses(out0, out1, gt):
    err0 = torch.mean(torch.abs(out0 - gt), dim=1, keepdim=True)
    err1 = torch.mean(torch.abs(out1 - gt), dim=1, keepdim=True)
    flow_loss = torch.minimum(err0, err1).mean()

    disagreement = torch.mean(torch.abs(out0 - out1), dim=1, keepdim=True)
    visible_weight = torch.exp(-8.0 * disagreement).detach()
    consistency_loss = (
        torch.abs(out0 - out1) * visible_weight
    ).mean()

    return flow_loss, consistency_loss

@torch.inference_mode()
def interpolate(model, f0, f1, mv, alpha=0.5):

    model.eval()

    if not torch.is_tensor(alpha):
        alpha = torch.tensor(
            [alpha],
            dtype=torch.float32,
            device=f0.device
        )

    alpha = alpha.view(-1,1,1,1)

    return model(f0, f1, mv, alpha)

# =========================================
# TRAIN
# =========================================
TRAIN_BATCH_SIZE = max(
    1,
    int(os.environ.get("TRAIN_BATCH_SIZE", "2" if LOW_RAM_MODE else "12"))
)
TRAIN_CLIPS_PER_EPOCH = max(
    1,
    int(os.environ.get(
        "TRAIN_CLIPS_PER_EPOCH",
        "24" if LOW_RAM_MODE else "64"
    ))
)
TRAIN_EPOCHS = 1
CLEAR_CUDA_CACHE = os.environ.get("CLEAR_CUDA_CACHE", "0") == "1"
EXPORT_WIDTH = int(os.environ.get("EXPORT_WIDTH", "480"))
EXPORT_HEIGHT = int(os.environ.get("EXPORT_HEIGHT", "270"))
EXPORT_STATIC_PATH = os.environ.get("EXPORT_STATIC_PATH", "ffgup_static.onnx")

def train(model, epochs=1):

    opt = optim.Adam(
        model.parameters(),
        lr = 1e-4
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_fp16
    )

    model.train()

    for e in range(epochs):

        total = 0
        count = 0
        batch_step = 0

        batches = AsyncBatchPrefetcher(
            lambda: generate_data(TRAIN_CLIPS_PER_EPOCH),
            TRAIN_BATCH_SIZE,
            PREFETCH_BATCHES,
        )

        for batch in batches:
        
            p0s, p1s, pts, alphas, mvs = zip(*batch)
            mv = torch.stack([
                torch.from_numpy(mv_patch).permute(2,0,1).float()
                for mv_patch in mvs
            ]).to(device, memory_format=torch.channels_last)
            
            mv = mv / 16.0
            t0 = torch.stack([
                np_to_tensor(p0)
                for p0 in p0s
            ]).to(device, memory_format=torch.channels_last)

            t1 = torch.stack([
                np_to_tensor(p1)
                for p1 in p1s
            ]).to(device, memory_format=torch.channels_last)

            tt = torch.stack([
                np_to_tensor(pt)
                for pt in pts
            ]).to(device, memory_format=torch.channels_last)
        
            a = torch.tensor(
                alphas,
                dtype=torch.float32,
                device=device
            ).view(-1,1,1,1)

            opt.zero_grad(set_to_none=True)

            autocast_device = (
                "cuda"
                if device == "cuda"
                else "cpu"
            )
            
            autocast_dtype = (
                torch.float16
                if device == "cuda"
                else torch.bfloat16
            )
            
            with torch.autocast(
                device_type=autocast_device,
                dtype=autocast_dtype,
                enabled=use_fp16
            ):
                out, out0, out1 = model(t0, t1, mv, a)
                if batch_step % TEMPORAL_CHECK_INTERVAL == 0:
                    reverse_mv = torch.flip(mv, dims=[-1]).clone()
                    reverse_mv[:, 0] *= -1.0
                    reverse_out, _, _ = model(t1, t0, reverse_mv, 1.0 - a)
                else:
                    reverse_mv = None
                    reverse_out = None
                tt2 = F.interpolate(
                   tt,
                   size=out.shape[-2:],
                   mode="bilinear",
                   align_corners=False
                )
                loss_flow_raw, loss_consistency = occlusion_aware_warp_losses(
                    out0,
                    out1,
                    tt
                )
                loss_flow = loss_flow_raw * OCCLUSION_FLOW_WEIGHT

                loss_l1 = F.l1_loss(
                    out,
                    tt2
                )
                
                loss_ssim = ssim_loss(
                    (out + 1.0) / 2.0,
                    (tt2 + 1.0) / 2.0
                )
                loss_temporal = (
                    F.l1_loss(out, reverse_out.detach())
                    if reverse_out is not None
                    else out.new_zeros(())
                )
                
                loss = (
                    1.0 * loss_l1 +
                    0.1 * loss_ssim +
                    loss_flow +
                    OCCLUSION_CONSISTENCY_WEIGHT * loss_consistency +
                    TEMPORAL_CONSISTENCY_WEIGHT * loss_temporal
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(batch)
            count += len(batch)
            batch_step += 1
        
            batch = []
            del p0s, p1s, pts, alphas, mvs
            del mv, reverse_mv, reverse_out, t0, t1, tt, a, out, out0, out1, tt2
            del loss_flow_raw, loss_flow, loss_consistency, loss_l1, loss_ssim, loss_temporal, loss
            if CLEAR_CUDA_CACHE and device == "cuda":
                torch.cuda.empty_cache()
        print(
            f"Epoch {e+1} "
            f"Loss {total / max(count, 1):.4f}"
        )

# =========================================
# TEST
# =========================================
@torch.inference_mode()
def test(model):

    model.eval()

    samples = list(generate_data(12))

    for batch_start in range(0, min(3, len(samples)), INFERENCE_BATCH_SIZE):
        batch = samples[batch_start:batch_start + INFERENCE_BATCH_SIZE]
        f0s, f1s, fts, alphas, mvs = zip(*batch)

        mv = torch.from_numpy(np.stack(mvs))\
            .permute(0,3,1,2)\
            .float()\
            .to(device, memory_format=torch.channels_last)
        
        mv = mv / 16.0

        t0 = torch.stack([np_to_tensor(frame) for frame in f0s]).to(
            device,
            memory_format=torch.channels_last
        )

        t1 = torch.stack([np_to_tensor(frame) for frame in f1s]).to(
            device,
            memory_format=torch.channels_last
        )
        
        gt = torch.stack([np_to_tensor(frame) for frame in fts]).to(
            device,
            memory_format=torch.channels_last
        )


        gt = F.interpolate(
            gt,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        gt = [tensor_to_np(target) for target in gt]
        
        a = torch.tensor(
            alphas,
            device=device
        ).view(-1,1,1,1)

        autocast_device = (
            "cuda"
            if device == "cuda"
            else "cpu"
        )

        autocast_dtype = (
            torch.float16
            if device == "cuda"
            else torch.bfloat16
        )

        with torch.autocast(
            device_type=autocast_device,
            dtype=autocast_dtype,
            enabled=use_fp16
        ):
            for _ in range(10):
                _ = model(t0, t1, mv, a)
            if device == "cuda":
                torch.cuda.synchronize()
            inicio = time.perf_counter()
            out, _, _ = model(t0, t1, mv, a)
            if device == "cuda":
                torch.cuda.synchronize()
            fin = time.perf_counter()

            latencia_ms = (fin - inicio) * 1000
            latencia_por_frame_ms = latencia_ms / len(batch)
            fps_equivalentes = 1000.0 / latencia_por_frame_ms
            print(
                f"Generacion intermedia ({len(batch)} frames): "
                f"{latencia_ms:.2f} ms | "
                f"{latencia_por_frame_ms:.2f} ms/frame | "
                f"{fps_equivalentes:.2f} FPS"
            )

        for offset, (prediction, target) in enumerate(zip(out, gt)):
            i = batch_start + offset
            pred = tensor_to_np(prediction)
            diff = diff_image(pred, target)

            cv2.imwrite(f"pred_{i}.png", pred)
            cv2.imwrite(f"gt_{i}.png", target)
            cv2.imwrite(f"diff_{i}.png", diff)

            psnr = peak_signal_noise_ratio(target, pred, data_range=255)
            ssim = structural_similarity(
                target,
                pred,
                channel_axis=2,
                data_range=255
            )

            print(f"PSNR: {psnr:.2f} dB")
            print(f"SSIM: {ssim:.4f}")
            print(f"guardado pred_{i}.png gt_{i}.png diff_{i}.png")


def quantize_onnx_to_int8(model_input, model_output):
    if not os.path.exists(model_input):
        print(f"No se puede cuantizar {model_input}: archivo no existe")
        return

    try:
        quantize_dynamic(
            model_input=model_input,
            model_output=model_output,
            weight_type=QuantType.QInt8,
            per_channel=False,
        )
        print(f"ONNX INT8: {model_output}")
    except Exception as exc:
        print(f"Error de cuantización INT8 para {model_input}: {exc}")


def export_onnx_models(model):
    model.eval()
    dynamic_height = 56
    dynamic_width = 56
    static_height = EXPORT_HEIGHT
    static_width = EXPORT_WIDTH

    dynamic_inputs = (
        torch.randn(1, 3, dynamic_height, dynamic_width, device=device),
        torch.randn(1, 3, dynamic_height, dynamic_width, device=device),
        torch.randn(1, 2, dynamic_height, dynamic_width, device=device),
        torch.tensor([[[[0.5]]]], device=device),
    )
    static_inputs = (
        torch.randn(1, 3, static_height, static_width, device=device),
        torch.randn(1, 3, static_height, static_width, device=device),
        torch.randn(1, 2, static_height, static_width, device=device),
        torch.tensor([[[[0.5]]]], device=device),
    )

    dynamic_path = "ffgup.onnx"
    static_path = EXPORT_STATIC_PATH

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dynamic_inputs,
            dynamic_path,
            input_names=["f0", "f1", "mv", "alpha"],
            output_names=["output"],
            dynamo=False,
            dynamic_axes={
                "f0": {2: "height", 3: "width"},
                "f1": {2: "height", 3: "width"},
                "mv": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            do_constant_folding=True,
            verbose=False,
        )

        torch.onnx.export(
            model,
            static_inputs,
            static_path,
            input_names=["f0", "f1", "mv", "alpha"],
            output_names=["output"],
            dynamo=False,
            do_constant_folding=True,
            verbose=False,
        )

    print(
        f"ONNX estatico: {static_path} "
        f"({EXPORT_WIDTH}x{EXPORT_HEIGHT})",
        flush=True,
    )

    if HAS_GPU:
        print("GPU detectada: generando variantes ONNX INT8")
        quantize_onnx_to_int8(dynamic_path, "ffgup_int8.onnx")
        quantize_onnx_to_int8(static_path, "ffgup_static_int8.onnx")
    else:
        print("Sin GPU detectada: se omite la cuantización INT8")

# =========================================
# RUN
# =========================================
if __name__ == "__main__":
    print("Device:", device)
    print("FP16:", use_fp16)
    print("AGGRESSIVE_PROFILE:", AGGRESSIVE_PROFILE)
    print(
        "LOW_RAM_MODE:",
        LOW_RAM_MODE,
        "FRAME_SIZE:",
        FRAME_SIZE,
        "SAMPLES_PER_CLIP:",
        SAMPLES_PER_CLIP,
        "FAST_MOTION:",
        FAST_MOTION,
        "BATCH:",
        TRAIN_BATCH_SIZE,
        "FRAME_CACHE:",
        FRAME_CACHE_SIZE,
        "MV_CACHE:",
        MV_CACHE_SIZE,
        "CLEAR_CUDA_CACHE:",
        CLEAR_CUDA_CACHE,
        "ANTI_GHOSTING:",
        ANTI_GHOSTING
    )

    model = FlowNet()\
        .to(device)\
        .to(memory_format=torch.channels_last)
    print(
        "Entrenamiento:",
        TRAIN_EPOCHS,
        "epocas |",
        TRAIN_CLIPS_PER_EPOCH,
        "clips/epoca"
    )
    train(model, epochs=TRAIN_EPOCHS)

    model.eval()

    try:
        export_onnx_models(model)
    except Exception as e:
        print("ONNX:", e)

    test(model)
