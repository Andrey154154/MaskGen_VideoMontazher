# -*- coding: utf-8 -*-
# ВидеоМонтажер — MaskGen
# BiRefNet (Hugging Face / локально) → Маска / Зелёный фон / Замена фона / Альфа (RGBA)
# Русский UI, пресеты качества, Soft Alpha, Alpha Power, Edge Shrink/Blur, Despill,
# прокси можно отключать, выбор кодеков (в т.ч. ProRes4444 alpha / PNG-секвенция RGBA), лог скорости.
# Дополнительно: копирование звука из исходника во всех режимах, кроме "Маска (ч/б)".
#  - Для видео-выводов звук добавляется в контейнер.
#  - Для "PNG-секвенция (RGBA)" создаётся WAV-файл <root>_audio.wav.

import os, sys, math, shutil, subprocess, time
import numpy as np
import cv2
from tqdm import tqdm

import torch
from transformers import AutoModelForImageSegmentation
from PySide6 import QtCore
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QComboBox
)

# --------- ffmpeg из папки проекта ./bin ---------
FFMPEG_BIN = os.path.join(
    os.path.dirname(__file__),
    "bin",
    "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
)

def _ensure_ffmpeg():
    if not os.path.isfile(FFMPEG_BIN):
        raise RuntimeError(f"ffmpeg не найден: {FFMPEG_BIN}")
    return FFMPEG_BIN

# --------- Пресеты ---------
PRESETS = {
    "Быстро (General)": {
        "model_id": "ZhengPeng7/BiRefNet-matting",
        "input_side": 1024,
        "proxy_on": True,
        "proxy_side": 1080,
        "tiles": 2,
        "overlap": 64,
        "soft_alpha": True,
        "alpha_power": 0.9,
        "edge_shrink": 0,
        "edge_blur": 3,
        "despill": True,
    },
    "Качество (HR)": {
        "model_id": "ZhengPeng7/BiRefNet_HR-matting",
        "input_side": 2048,
        "proxy_on": True,
        "proxy_side": 1440,
        "tiles": 2,
        "overlap": 96,
        "soft_alpha": True,
        "alpha_power": 0.9,
        "edge_shrink": 1,
        "edge_blur": 4,
        "despill": True,
    },
    "Макс. детали (медленно)": {
        "model_id": "ZhengPeng7/BiRefNet_HR-matting",
        "input_side": 2048,
        "proxy_on": False,     # без прокси
        "proxy_side": 0,
        "tiles": 2,
        "overlap": 128,
        "soft_alpha": True,
        "alpha_power": 0.85,
        "edge_shrink": 1,
        "edge_blur": 5,
        "despill": True,
    },
}

# --------- Вспомогательные ---------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def letterbox_pad(h, w, side):
    scale = side / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    top = (side - nh) // 2
    bottom = side - nh - top
    left = (side - nw) // 2
    right = side - nw - left
    return scale, top, left, nh, nw, bottom, right

def maybe_proxy(frame: np.ndarray, use_proxy: bool, max_side: int):
    h, w = frame.shape[:2]
    if (not use_proxy) or max(h, w) <= max_side:
        return frame, (h, w)
    sc = max_side / float(max(h, w))
    nw, nh = int(round(w * sc)), int(round(h * sc))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), (h, w)

def ffmpeg_supports_nvenc() -> bool:
    try:
        out = subprocess.check_output([_ensure_ffmpeg(), '-hide_banner', '-encoders'],
                                      stderr=subprocess.STDOUT, text=True)
        return ('h264_nvenc' in out) or ('hevc_nvenc' in out)
    except Exception:
        return False

def is_mov(path:str) -> bool:
    return os.path.splitext(path)[1].lower() == '.mov'

def build_ffmpeg_cmd(path: str, width: int, height: int, fps: float, codec: str, pix_fmt_in: str, audio_path: str|None):
    base = [
        _ensure_ffmpeg(), '-y',
        '-f', 'rawvideo', '-pix_fmt', pix_fmt_in, '-s', f'{width}x{height}',
        '-r', f'{fps}', '-i', '-'
    ]
    # Видеокодеки
    if codec == 'H264 (NVENC)':
        v = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-b:v', '0', '-cq', '19']
    elif codec == 'H264 (libx264)':
        v = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15']
    elif codec == 'HEVC (NVENC)':
        v = ['-c:v', 'hevc_nvenc', '-preset', 'p5', '-b:v', '0', '-cq', '19']
    elif codec == 'HEVC (libx265)':
        v = ['-c:v', 'libx265', '-pix_fmt', 'yuv420p', '-crf', '18']
    elif codec == 'MJPEG':
        v = ['-c:v', 'mjpeg', '-qscale:v', '3']
    elif codec == 'RAW':
        v = ['-c:v', 'rawvideo']
        if not path.lower().endswith('.avi'):
            root, _ = os.path.splitext(path)
            path = root + '.avi'
    else:
        v = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '15']

    cmd = base[:]
    # Аудио как второй вход
    if audio_path:
        cmd += ['-i', audio_path]
        # Маппинг и кодек аудио по контейнеру
        if is_mov(path) or codec == 'ProRes 4444 (MOV, alpha)':
            a = ['-c:a', 'pcm_s16le']
        else:
            a = ['-c:a', 'aac', '-b:a', '192k']
        cmd += v + a + ['-map', '0:v:0', '-map', '1:a:0', '-shortest', path]
    else:
        cmd += v + [path]
    return cmd, path

def extract_audio_wav(src: str, wav_path: str):
    # Извлечь исходный звук в WAV (PCM 16-bit)
    subprocess.run([
        _ensure_ffmpeg(), '-y', '-i', src, '-vn', '-acodec', 'pcm_s16le', wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class FFmpegWriter:
    def __init__(self, path: str, width: int, height: int, fps: float, codec: str, color: bool, audio_path: str|None):
        self.width = int(width); self.height = int(height)
        self.fps = float(fps) if fps and not math.isnan(fps) else 25.0
        _ensure_ffmpeg()
        pix_fmt_in = 'bgr24' if color else 'gray8'
        self.cmd, self.path = build_ffmpeg_cmd(path, width, height, self.fps, codec, pix_fmt_in, audio_path)
        enc_name = None
        if '-c:v' in self.cmd:
            try:
                enc_name = self.cmd[self.cmd.index('-c:v')+1]
            except Exception:
                enc_name = 'libx264'
        print(f"[ffmpeg] {os.path.basename(self.path)} ← {enc_name} {self.width}x{self.height}@{self.fps} | pix_in={pix_fmt_in} | audio={'ON' if audio_path else 'OFF'}")
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.closed = False
    def write_gray(self, gray_u8: np.ndarray):
        if gray_u8.ndim != 2:
            raise ValueError("Ожидается HxW (GRAY8)")
        if gray_u8.shape[:2] != (self.height, self.width):
            gray_u8 = cv2.resize(gray_u8, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self.proc.stdin.write(gray_u8.tobytes())
    def write_bgr(self, bgr_u8: np.ndarray):
        if bgr_u8.ndim != 3 or bgr_u8.shape[2] != 3:
            raise ValueError("Ожидается HxWx3 (BGR)")
        if bgr_u8.shape[:2] != (self.height, self.width):
            bgr_u8 = cv2.resize(bgr_u8, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self.proc.stdin.write(bgr_u8.tobytes())
    def release(self):
        if not self.closed:
            try:
                self.proc.stdin.flush(); self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait(timeout=10)
            self.closed = True

# ---------- Альфа-варианты вывода ----------
class FFmpegAlphaWriter:
    """Пишет видео с альфой в ProRes 4444 (MOV) через pipe raw BGRA. Добавляет аудио из исходника."""
    def __init__(self, path: str, width: int, height: int, fps: float, audio_path: str|None):
        self.width, self.height = int(width), int(height)
        self.fps = float(fps) if fps and not math.isnan(fps) else 25.0
        _ensure_ffmpeg()
        self.cmd = [
            _ensure_ffmpeg(), '-y',
            '-f','rawvideo','-pix_fmt','bgra','-s',f'{self.width}x{self.height}',
            '-r',f'{self.fps}','-i','-'
        ]
        if audio_path:
            self.cmd += ['-i', audio_path]
        self.cmd += [
            '-c:v','prores_ks','-profile:v','4444','-pix_fmt','yuva444p10le'
        ]
        if audio_path:
            self.cmd += ['-c:a','pcm_s16le', '-map','0:v:0','-map','1:a:0','-shortest']
        self.cmd += [path]

        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.closed = False
    def write_bgra(self, bgra_u8: np.ndarray):
        if bgra_u8.shape[:2] != (self.height, self.width):
            bgra_u8 = cv2.resize(bgra_u8, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self.proc.stdin.write(bgra_u8.tobytes())
    def release(self):
        if not self.closed:
            try:
                self.proc.stdin.flush(); self.proc.stdin.close()
            except Exception: 
                pass
            self.proc.wait(timeout=10)
            self.closed = True

class PNGSequenceWriter:
    """Пишет RGBA PNG-секвенцию (ожидает BGRA на входе, OpenCV сам корректно сохранит alpha)."""
    def __init__(self, out_dir: str, base: str):
        self.dir = out_dir; os.makedirs(self.dir, exist_ok=True)
        self.base = base
        self.i = 0
    def write_bgra(self, bgra_u8: np.ndarray):
        fn = os.path.join(self.dir, f"{self.base}_{self.i:06d}.png")
        cv2.imwrite(fn, bgra_u8)  # BGRA → PNG с альфой
        self.i += 1
    def release(self):
        pass

# --------- Модель ---------
class HF_BiRefNet:
    def __init__(self, model_id_or_dir: str, input_side: int = 2048, prefer_cuda_fp16: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.side = int(input_side)
        self.fp16 = bool(prefer_cuda_fp16 and self.device.type == 'cuda')

        is_local_dir = os.path.isdir(model_id_or_dir) and os.path.isfile(os.path.join(model_id_or_dir, 'config.json'))
        print(f"[HF] загрузка {'локальной папки модели' if is_local_dir else 'HF model_id'}: {model_id_or_dir}")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id_or_dir, trust_remote_code=True
        )
        self.model.to(self.device).eval()
        if self.fp16:
            self.model.half()
        print(f"[HF] device={self.device}, fp16={'ДА' if self.fp16 else 'НЕТ'}")

        self.mean = IMAGENET_MEAN.to(self.device)
        self.std  = IMAGENET_STD .to(self.device)
        if self.fp16:
            self.mean = self.mean.half(); self.std = self.std.half()

    @torch.no_grad()
    def infer_one(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _, top, left, nh, nw, bottom, right = letterbox_pad(h, w, self.side)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        t = torch.from_numpy(pad).to(self.device)  # H,W,3
        t = t.permute(2,0,1).unsqueeze(0)          # 1,3,H,W
        t = t.to(torch.float16 if self.fp16 else torch.float32) / 255.0
        t = (t - self.mean) / self.std

        if self.fp16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = self.model(t)
        else:
            out = self.model(t)

        logits = out.get('logits', None) if isinstance(out, dict) else out
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if logits.shape[1] > 1:
            logits = logits[:,1:2]  # эвристика: foreground

        prob = torch.sigmoid(logits)[0,0]
        prob = prob[top:top+nh, left:left+nw]
        prob = prob.float().clamp(0,1).detach().cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return (prob * 255.0).astype(np.uint8)

    @torch.no_grad()
    def infer_tiled(self, frame_bgr: np.ndarray, tiles: int = 2, overlap: int = 64) -> np.ndarray:
        H, W = frame_bgr.shape[:2]
        grid = max(1, int(tiles))
        if grid == 1:
            return self.infer_one(frame_bgr)
        tw, th = W // grid, H // grid
        acc = np.zeros((H, W), dtype=np.float32)
        wgt = np.zeros((H, W), dtype=np.float32)
        for gy in range(grid):
            for gx in range(grid):
                x0 = max(0, gx*tw - overlap); y0 = max(0, gy*th - overlap)
                x1 = min(W, (gx+1)*tw + overlap); y1 = min(H, (gy+1)*th + overlap)
                crop = frame_bgr[y0:y1, x0:x1]
                m = self.infer_one(crop).astype(np.float32) / 255.0
                wy = np.hanning(m.shape[0]) if m.shape[0] > 1 else np.ones(1)
                wx = np.hanning(m.shape[1]) if m.shape[1] > 1 else np.ones(1)
                w2 = np.outer(wy, wx).astype(np.float32)
                acc[y0:y1, x0:x1] += m * w2
                wgt[y0:y1, x0:x1] += w2
        wgt[wgt == 0] = 1.0
        out = (acc / wgt)
        return (out * 255.0).clip(0,255).astype(np.uint8)

# --------- Фон ---------
class BGProvider:
    def __init__(self, mode: str, solid_bgr=(0,255,0), bg_path: str|None=None, loop: bool=True):
        self.mode = mode  # 'mask' | 'green' | 'replace'
        self.color = tuple(int(x) for x in solid_bgr)
        self.path = bg_path
        self.loop = loop
        self.cap = None
        self.bg_img = None
        if self.mode == 'replace' and self.path:
            ext = os.path.splitext(self.path)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
                img = cv2.imread(self.path, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Не удалось открыть фон: {self.path}")
                self.bg_img = img
            else:
                self.cap = cv2.VideoCapture(self.path)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Не удалось открыть видео фон: {self.path}")

    def get(self, W: int, H: int, frame_idx: int) -> np.ndarray:
        if self.mode == 'green':
            return np.full((H, W, 3), self.color, dtype=np.uint8)
        if self.mode == 'replace':
            if self.bg_img is not None:
                return cv2.resize(self.bg_img, (W, H), interpolation=cv2.INTER_AREA)
            if self.cap is not None:
                ret, frm = self.cap.read()
                if not ret:
                    if self.loop:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frm = self.cap.read()
                        if not ret:
                            return np.zeros((H, W, 3), np.uint8)
                    else:
                        return np.zeros((H, W, 3), np.uint8)
                if frm.shape[1] != W or frm.shape[0] != H:
                    frm = cv2.resize(frm, (W, H), interpolation=cv2.INTER_AREA)
                return frm
        return np.zeros((H, W, 3), np.uint8)

    def release(self):
        if self.cap is not None:
            self.cap.release()

# --------- Воркёр ---------
class VideoWorker(QtCore.QObject):
    progress = QtCore.Signal(int, int)
    message = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self, model_path_or_id: str, input_side: int,
        tiles: int, overlap: int,
        use_proxy: bool, proxy_max: int,
        codec: str, alpha_codec: str, src: str, dst: str,
        out_mode: str, solid_bgr: tuple[int,int,int],
        bg_path: str|None, loop_bg: bool,
        soft_alpha: bool, alpha_power: float,
        edge_shrink: int, edge_blur: int, despill: bool,
        threshold: float,
        copy_audio: bool
    ):
        super().__init__()
        self.model_path_or_id = model_path_or_id
        self.input_side = input_side
        self.tiles = tiles
        self.overlap = overlap
        self.use_proxy = use_proxy
        self.proxy_max = proxy_max
        self.codec = codec
        self.alpha_codec = alpha_codec
        self.src = src
        self.dst = dst
        self.out_mode = out_mode
        self.solid_bgr = solid_bgr
        self.bg_path = bg_path
        self.loop_bg = loop_bg
        self.soft_alpha = soft_alpha
        self.alpha_power = alpha_power
        self.edge_shrink = max(0, edge_shrink)
        self.edge_blur = max(0, edge_blur)
        self.despill = despill
        self.threshold = threshold
        self.copy_audio = copy_audio

    @QtCore.Slot()
    def run(self):
        try:
            backend = HF_BiRefNet(self.model_path_or_id, input_side=self.input_side, prefer_cuda_fp16=True)
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                raise RuntimeError(f"Не удалось открыть видео: {self.src}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            # Режимы вывода
            color_output = (self.out_mode != 'mask' and self.out_mode != 'alpha')
            include_audio = (self.out_mode != 'mask') and self.copy_audio

            # Инициализация writer
            if self.out_mode == 'alpha':
                if 'PNG-секвенция' in self.alpha_codec:
                    outdir = os.path.splitext(self.dst)[0] + "_rgba_seq"
                    base = os.path.splitext(os.path.basename(self.dst))[0]
                    writer = PNGSequenceWriter(outdir, base)
                    # параллельно извлечь WAV, если нужен звук
                    if include_audio:
                        wav_path = os.path.splitext(self.dst)[0] + "_audio.wav"
                        extract_audio_wav(self.src, wav_path)
                        self.message.emit(f"Звук сохранён → {wav_path}")
                else:
                    # Принудительно MOV
                    root, _ = os.path.splitext(self.dst)
                    self.dst = root + "_alpha.mov"
                    writer = FFmpegAlphaWriter(self.dst, W, H, fps, audio_path=(self.src if include_audio else None))
            else:
                chosen_codec = self.codec
                if chosen_codec in ('ProRes 4444 (MOV, alpha)', 'PNG-секвенция (RGBA)'):
                    chosen_codec = 'H264 (libx264)'
                    self.message.emit("Выбран альфа-кодек вне режима 'Альфа-канал'. Использую H264 (libx264).")
                writer = FFmpegWriter(self.dst, W, H, fps, codec=chosen_codec, color=color_output, audio_path=(self.src if include_audio else None))

            self.message.emit(
                f"Старт: {total} кадров, {W}x{H} @ {fps:.2f} | устройство={backend.device}, fp16={'ДА' if backend.fp16 else 'НЕТ'} | звук={'ДА' if include_audio else 'НЕТ'}"
            )

            bg = BGProvider(
                'green' if self.out_mode=='green' else ('replace' if self.out_mode=='replace' else 'mask'),
                solid_bgr=self.solid_bgr, bg_path=self.bg_path, loop=self.loop_bg
            )

            pbar = tqdm(total=total, desc=("Альфа (RGBA)" if self.out_mode=='alpha' else ("Композит" if color_output else "Маска")), unit="кадр")
            processed, t0 = 0, time.time()

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    pframe, (oh, ow) = maybe_proxy(frame, self.use_proxy, self.proxy_max)

                    # маска как вероятности [0..255]
                    mask_u8 = backend.infer_tiled(pframe, tiles=self.tiles, overlap=self.overlap) \
                              if self.tiles>1 else backend.infer_one(pframe)

                    if (mask_u8.shape[0], mask_u8.shape[1]) != (oh, ow):
                        mask_u8 = cv2.resize(mask_u8, (ow, oh), interpolation=cv2.INTER_LINEAR)

                    # Подготовка альфы (общая для композита и режима 'alpha')
                    prob = (mask_u8.astype(np.float32) / 255.0)
                    if self.edge_shrink > 0:
                        k = self.edge_shrink*2+1
                        prob = cv2.erode(prob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
                    if self.edge_blur > 0:
                        k = self.edge_blur if self.edge_blur % 2 == 1 else self.edge_blur + 1
                        prob = cv2.GaussianBlur(prob, (k, k), 0)
                    if self.soft_alpha:
                        alpha_f = np.clip(prob ** float(self.alpha_power), 0.0, 1.0)
                    else:
                        alpha_f = (prob >= float(self.threshold)).astype(np.float32)

                    if self.out_mode == 'alpha':
                        # Создаём BGRA: RGB из исходника (с учётом прокси), A — из alpha_f
                        if (oh, ow) != (H, W):
                            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
                        a = (alpha_f * 255.0).clip(0,255).astype(np.uint8)
                        bgra = np.dstack([pframe, a])  # B,G,R,A
                        if (oh, ow) != (H, W):
                            bgra = cv2.resize(bgra, (W, H), interpolation=cv2.INTER_NEAREST)
                        writer.write_bgra(bgra)
                    elif not color_output:
                        mask_out = (alpha_f * 255.0).astype(np.uint8) if self.soft_alpha else (alpha_f.astype(np.uint8) * 255)
                        writer.write_gray(mask_out)
                    else:
                        if (oh, ow) != (H, W):
                            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
                        alpha = alpha_f[..., None]
                        bg_bgr = bg.get(ow, oh, processed)
                        comp = (pframe.astype(np.float32) * alpha +
                                bg_bgr.astype(np.float32) * (1.0 - alpha))
                        if self.despill:
                            edge_w = np.clip(cv2.Laplacian(alpha_f, cv2.CV_32F), 0, 1)
                            edge_w = cv2.GaussianBlur(edge_w, (5,5), 0)[..., None]
                            comp = comp*(1 - 0.08*edge_w) + bg_bgr.astype(np.float32)*(0.08*edge_w)
                        comp = comp.clip(0,255).astype(np.uint8)
                        if (oh, ow) != (H, W):
                            comp = cv2.resize(comp, (W, H), interpolation=cv2.INTER_LINEAR)
                        writer.write_bgr(comp)

                    processed += 1; pbar.update(1)
                    if processed % 5 == 0:
                        dt = (time.time() - t0) / max(1, processed)
                        self.message.emit(f"{processed}/{total} | {dt:.3f} сек/кадр")
                    self.progress.emit(processed, total)

            finally:
                pbar.close(); cap.release(); writer.release(); bg.release()

        except Exception as e:
            self.message.emit(f"Ошибка: {e}")
        finally:
            self.finished.emit()

# --------- GUI ---------
class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ВидеоМонтажер — MaskGen")
        self.setMinimumWidth(1040)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        # --- Источник видео ---
        gb_src = QGroupBox("Источник видео")
        s = QGridLayout(gb_src)
        self.inp = QLineEdit(); b_in = QPushButton("Выбрать…"); b_in.clicked.connect(self._pick_in)
        self.out_dir = QLineEdit("masks"); b_out = QPushButton("Папка вывода…"); b_out.clicked.connect(self._pick_out)
        s.addWidget(QLabel("Файл:"), 0, 0); s.addWidget(self.inp, 0, 1, 1, 3); s.addWidget(b_in, 0, 4)
        s.addWidget(QLabel("Папка вывода:"), 1, 0); s.addWidget(self.out_dir, 1, 1, 1, 3); s.addWidget(b_out, 1, 4)
        lay.addWidget(gb_src)

        # --- Модель ---
        gb_m = QGroupBox("Модель (Hugging Face или локальная папка)")
        m = QGridLayout(gb_m)
        self.preset = QComboBox(); self.preset.addItems(list(PRESETS.keys())); self.preset.currentTextChanged.connect(self._apply_preset)
        self.use_local = QCheckBox("Использовать локальную папку модели")
        self.model_id = QLineEdit(PRESETS["Качество (HR)"]["model_id"])
        self.local_dir = QLineEdit(); b_loc = QPushButton("Папка…"); b_loc.clicked.connect(self._pick_local_dir)
        self.input_side = QSpinBox(); self.input_side.setRange(256, 4096); self.input_side.setValue(PRESETS["Качество (HR)"]["input_side"])
        m.addWidget(QLabel("Пресет:"), 0, 0); m.addWidget(self.preset, 0, 1)
        m.addWidget(self.use_local, 0, 2, 1, 2)
        m.addWidget(QLabel("HF model_id:"), 1, 0); m.addWidget(self.model_id, 1, 1, 1, 3)
        m.addWidget(QLabel("Локальная папка:"), 2, 0); m.addWidget(self.local_dir, 2, 1, 1, 2); m.addWidget(b_loc, 2, 3)
        m.addWidget(QLabel("Размер входа (px):"), 3, 0); m.addWidget(self.input_side, 3, 1)
        lay.addWidget(gb_m)

        # --- Режим вывода ---
        gb_mode = QGroupBox("Режим вывода")
        mo = QGridLayout(gb_mode)
        self.mode = QComboBox(); 
        self.mode.addItems([
            "Маска (ч/б)",
            "Зелёный фон (однотонный)",
            "Замена фона (картинка/видео)",
            "Альфа-канал (RGBA)"
        ])
        self.bg_path = QLineEdit(); b_bg = QPushButton("Фон…"); b_bg.clicked.connect(self._pick_bg)
        self.loop_bg = QCheckBox("Зацикливать фон"); self.loop_bg.setChecked(True)
        self.b = QSpinBox(); self.b.setRange(0,255); self.b.setValue(0)
        self.g = QSpinBox(); self.g.setRange(0,255); self.g.setValue(255)
        self.r = QSpinBox(); self.r.setRange(0,255); self.r.setValue(0)
        mo.addWidget(QLabel("Режим:"), 0, 0); mo.addWidget(self.mode, 0, 1, 1, 3)
        mo.addWidget(QLabel("Фон (картинка/видео):"), 1, 0); mo.addWidget(self.bg_path, 1, 1, 1, 2); mo.addWidget(b_bg, 1, 3); mo.addWidget(self.loop_bg, 1, 4)
        mo.addWidget(QLabel("Цвет фона (B,G,R):"), 2, 0); mo.addWidget(self.b, 2, 1); mo.addWidget(self.g, 2, 2); mo.addWidget(self.r, 2, 3)
        self.mode.currentTextChanged.connect(self._mode_changed)
        lay.addWidget(gb_mode)

        # --- Качество и композит ---
        gb_q = QGroupBox("Качество и композит")
        q = QGridLayout(gb_q)
        # Инференс
        self.tiles = QSpinBox(); self.tiles.setRange(1, 4); self.tiles.setValue(PRESETS["Качество (HR)"]["tiles"])
        self.overlap = QSpinBox(); self.overlap.setRange(0, 256); self.overlap.setValue(PRESETS["Качество (HR)"]["overlap"])
        self.use_proxy = QCheckBox("Использовать прокси"); self.use_proxy.setChecked(PRESETS["Качество (HR)"]["proxy_on"])
        self.proxy = QSpinBox(); self.proxy.setRange(256, 8192); self.proxy.setValue(PRESETS["Качество (HR)"]["proxy_side"])
        self.use_proxy.toggled.connect(lambda v: self.proxy.setEnabled(v))
        # Альфа/край
        self.soft_alpha = QCheckBox("Soft Alpha (без порога)"); self.soft_alpha.setChecked(PRESETS["Качество (HR)"]["soft_alpha"])
        self.alpha_power = QDoubleSpinBox(); self.alpha_power.setRange(0.5, 1.5); self.alpha_power.setSingleStep(0.05); self.alpha_power.setValue(PRESETS["Качество (HR)"]["alpha_power"])
        self.edge_shrink = QSpinBox(); self.edge_shrink.setRange(0, 10); self.edge_shrink.setValue(PRESETS["Качество (HR)"]["edge_shrink"])
        self.edge_blur = QSpinBox(); self.edge_blur.setRange(0, 41); self.edge_blur.setValue(PRESETS["Качество (HR)"]["edge_blur"])
        self.despill = QCheckBox("Despill (приглушать ореол)"); self.despill.setChecked(PRESETS["Качество (HR)"]["despill"])
        # Порог (на случай если Soft Alpha выключен)
        self.thr = QDoubleSpinBox(); self.thr.setRange(0.0,1.0); self.thr.setSingleStep(0.01); self.thr.setValue(0.5)

        q.addWidget(QLabel("Тайлы (NxN):"), 0, 0); q.addWidget(self.tiles, 0, 1)
        q.addWidget(QLabel("Перекрытие (px):"), 0, 2); q.addWidget(self.overlap, 0, 3)
        q.addWidget(self.use_proxy, 1, 0); q.addWidget(QLabel("Прокси, макс. сторона:"), 1, 2); q.addWidget(self.proxy, 1, 3)

        q.addWidget(self.soft_alpha, 2, 0); q.addWidget(QLabel("Alpha Power (γ):"), 2, 2); q.addWidget(self.alpha_power, 2, 3)
        q.addWidget(QLabel("Сжать край (px):"), 3, 0); q.addWidget(self.edge_shrink, 3, 1)
        q.addWidget(QLabel("Размыть край (px):"), 3, 2); q.addWidget(self.edge_blur, 3, 3)
        q.addWidget(self.despill, 4, 0); q.addWidget(QLabel("Порог (если без Soft Alpha):"), 4, 2); q.addWidget(self.thr, 4, 3)

        lay.addWidget(gb_q)

        # --- Кодек и сохранение ---
        gb_c = QGroupBox("Кодек и сохранение")
        c = QGridLayout(gb_c)
        self.codec = QComboBox()
        self.codec.addItems([
            'H264 (NVENC)', 'H264 (libx264)',
            'HEVC (NVENC)', 'HEVC (libx265)',
            'MJPEG', 'RAW',
            'ProRes 4444 (MOV, alpha)',
            'PNG-секвенция (RGBA)'
        ])
        if not ffmpeg_supports_nvenc():
            self.codec.setCurrentText('H264 (libx264)')
        self.copy_audio = QCheckBox("Копировать звук из исходника"); self.copy_audio.setChecked(True)
        c.addWidget(QLabel("Кодек / формат:"), 0, 0); c.addWidget(self.codec, 0, 1)
        c.addWidget(self.copy_audio, 1, 0, 1, 2)
        lay.addWidget(gb_c)

        # --- Кнопки ---
        hb_btn = QHBoxLayout()
        self.btn_run = QPushButton("Старт"); self.btn_run.clicked.connect(self._run)
        self.btn_prev = QPushButton("Предпросмотр (8 кадров)"); self.btn_prev.clicked.connect(self._preview)
        # self.btn_deps = QPushButton("Установить зависимости"); self.btn_deps.clicked.connect(self._install_deps)
        hb_btn.addWidget(self.btn_run); hb_btn.addWidget(self.btn_prev); hb_btn.addStretch(1);# hb_btn.addWidget(self.btn_deps)
        lay.addLayout(hb_btn)

        # --- Прогресс и лог ---
        self.prog = QProgressBar(); self.prog.setRange(0, 100)
        self.log = QLineEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.prog); lay.addWidget(self.log)

        # Инициализация
        self._apply_preset(self.preset.currentText())
        self._mode_changed(self.mode.currentText())

    # ---------- UI handlers ----------
    def _apply_preset(self, name: str):
        pr = PRESETS.get(name); 
        if not pr: return
        if not self.use_local.isChecked():
            self.model_id.setText(pr["model_id"])
        self.input_side.setValue(pr["input_side"])
        self.use_proxy.setChecked(pr["proxy_on"])
        self.proxy.setEnabled(pr["proxy_on"])
        self.proxy.setValue(pr["proxy_side"])
        self.tiles.setValue(pr["tiles"])
        self.overlap.setValue(pr["overlap"])
        self.soft_alpha.setChecked(pr["soft_alpha"])
        self.alpha_power.setValue(pr["alpha_power"])
        self.edge_shrink.setValue(pr["edge_shrink"])
        self.edge_blur.setValue(pr["edge_blur"])
        self.despill.setChecked(pr["despill"])
        self._log(f"Применён пресет: {name}")

    def _mode_changed(self, m: str):
        need_bg = ("Замена фона" in m)
        self.bg_path.setEnabled(need_bg)

    def _pick_in(self):
        p, _ = QFileDialog.getOpenFileName(self, "Выбор видео", "", "Видео (*.mp4 *.mov *.avi *.mkv)")
        if p: self.inp.setText(p)
    def _pick_out(self):
        p = QFileDialog.getExistingDirectory(self, "Папка вывода", "")
        if p: self.out_dir.setText(p)
    def _pick_local_dir(self):
        p = QFileDialog.getExistingDirectory(self, "Папка модели (config.json + model.safetensors)", "")
        if p: self.local_dir.setText(p)
    def _pick_bg(self):
        p, _ = QFileDialog.getOpenFileName(self, "Фон (картинка/видео)", "", "Медиа (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.mov *.avi *.mkv)")
        if p: self.bg_path.setText(p)

    def _dst(self, src: str):
        base = os.path.splitext(os.path.basename(src))[0]
        outdir = self.out_dir.text().strip() or "masks"
        os.makedirs(outdir, exist_ok=True)
        return os.path.join(outdir, f"{base}_out.mp4")

    def _install_deps(self):
        pkgs = ["torch", "torchvision", "torchaudio", "transformers", "safetensors", "timm", "kornia", "einops", "opencv-python", "tqdm", "PySide6"]
        self._log("Установка зависимостей…")
        try:
            out = subprocess.check_output([sys.executable, "-m", "pip", "install", "-U"] + pkgs,
                                          stderr=subprocess.STDOUT, text=True)
            self._log("Завершено")
        except subprocess.CalledProcessError as e:
            self._log("pip error: " + (e.output.splitlines()[-1] if e.output else str(e)))

    def _run(self):
        src = self.inp.text().strip()
        if not os.path.isfile(src):
            return self._log("Не выбран входной файл")
        model_path_or_id = self.local_dir.text().strip() if self.use_local.isChecked() and self.local_dir.text().strip() else self.model_id.text().strip()
        if "Замена фона" in self.mode.currentText():
            if not os.path.isfile(self.bg_path.text().strip()):
                return self._log("Укажите фон (картинка/видео)")
        dst = self._dst(src)
        self._log("Обработка…")
        self.prog.setValue(0)

        mode_txt = self.mode.currentText()
        out_mode = (
            'mask' if 'Маска' in mode_txt else
            ('green' if 'Зелёный' in mode_txt else
             ('alpha' if 'Альфа-канал' in mode_txt else 'replace'))
        )

        self.thread = QtCore.QThread(self)
        self.worker = VideoWorker(
            model_path_or_id=model_path_or_id,
            input_side=int(self.input_side.value()),
            tiles=int(self.tiles.value()),
            overlap=int(self.overlap.value()),
            use_proxy=self.use_proxy.isChecked(),
            proxy_max=int(self.proxy.value()),
            codec=self.codec.currentText(),
            alpha_codec=self.codec.currentText(),
            src=src, dst=dst,
            out_mode=out_mode,
            solid_bgr=(self.b.value(), self.g.value(), self.r.value()),
            bg_path=self.bg_path.text().strip() if out_mode=='replace' else None,
            loop_bg=self.loop_bg.isChecked(),
            soft_alpha=self.soft_alpha.isChecked(),
            alpha_power=float(self.alpha_power.value()),
            edge_shrink=int(self.edge_shrink.value()),
            edge_blur=int(self.edge_blur.value()),
            despill=self.despill.isChecked(),
            threshold=float(self.thr.value()),
            copy_audio=self.copy_audio.isChecked()
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.btn_run.setEnabled(False); self.btn_prev.setEnabled(False)
        self.thread.start()

    def _on_progress(self, processed: int, total: int):
        pct = int(processed / max(1, total) * 100)
        self.prog.setValue(min(100, pct))
    def _on_finished(self):
        self.btn_run.setEnabled(True); self.btn_prev.setEnabled(True)
        if self.prog.value() < 100: self.prog.setValue(100)
        self._log("Готово")

    def _preview(self):
        # Быстрый предпросмотр (8 кадров). Для режима "Альфа-канал" показываем исходник + альфу.
        src = self.inp.text().strip()
        if not os.path.isfile(src):
            return self._log("Не выбран входной файл")
        model_path_or_id = self.local_dir.text().strip() if self.use_local.isChecked() and self.local_dir.text().strip() else self.model_id.text().strip()
        mode_txt = self.mode.currentText()
        out_mode = (
            'mask' if 'Маска' in mode_txt else
            ('green' if 'Зелёный' in mode_txt else
             ('alpha' if 'Альфа-канал' in mode_txt else 'replace'))
        )
        if out_mode == 'replace' and not os.path.isfile(self.bg_path.text().strip()):
            return self._log("Укажите фон (картинка/видео)")

        try:
            backend = HF_BiRefNet(model_path_or_id, input_side=int(self.input_side.value()), prefer_cuda_fp16=True)
        except Exception as e:
            return self._log(f"Инициализация модели: {e}")

        cap = cv2.VideoCapture(src)
        frames = []
        for _ in range(8):
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()
        if not frames:
            return self._log("Нет кадров для предпросмотра")

        bg = BGProvider(
            'green' if out_mode=='green' else ('replace' if out_mode=='replace' else 'mask'),
            solid_bgr=(self.b.value(), self.g.value(), self.r.value()),
            bg_path=self.bg_path.text().strip() if out_mode=='replace' else None,
            loop=self.loop_bg.isChecked()
        )

        rows = []
        for i, f in enumerate(frames):
            pframe, (oh, ow) = maybe_proxy(f, self.use_proxy.isChecked(), int(self.proxy.value()))
            mask_u8 = backend.infer_tiled(pframe, tiles=int(self.tiles.value()), overlap=int(self.overlap.value())) \
                      if int(self.tiles.value())>1 else backend.infer_one(pframe)
            if (mask_u8.shape[0], mask_u8.shape[1]) != (oh, ow):
                mask_u8 = cv2.resize(mask_u8, (ow, oh), interpolation=cv2.INTER_LINEAR)

            prob = (mask_u8.astype(np.float32)/255.0)
            if int(self.edge_shrink.value()) > 0:
                k = int(self.edge_shrink.value())*2+1
                prob = cv2.erode(prob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
            if int(self.edge_blur.value()) > 0:
                k = int(self.edge_blur.value())
                if k % 2 == 0: k += 1
                prob = cv2.GaussianBlur(prob, (k,k), 0)
            if self.soft_alpha.isChecked():
                alpha_f = np.clip(prob ** float(self.alpha_power.value()), 0.0, 1.0)
            else:
                alpha_f = (prob >= float(self.thr.value())).astype(np.float32)

            if out_mode == 'mask':
                mask_vis = (alpha_f * 255.0).astype(np.uint8)
                vis = np.hstack([
                    cv2.resize(pframe, (pframe.shape[1]//2, pframe.shape[0]//2)),
                    cv2.resize(cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR), (pframe.shape[1]//2, pframe.shape[0]//2))
                ])
            elif out_mode == 'alpha':
                mask_vis = (alpha_f * 255.0).astype(np.uint8)
                vis = np.hstack([
                    cv2.resize(pframe, (pframe.shape[1]//2, pframe.shape[0]//2)),
                    cv2.resize(cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR), (pframe.shape[1]//2, pframe.shape[0]//2))
                ])
            else:
                alpha = alpha_f[..., None]
                bg_bgr = bg.get(ow, oh, i)
                comp = (pframe.astype(np.float32) * alpha + bg_bgr.astype(np.float32) * (1.0 - alpha))
                if self.despill.isChecked():
                    edge_w = np.clip(cv2.Laplacian(alpha_f, cv2.CV_32F), 0, 1)
                    edge_w = cv2.GaussianBlur(edge_w, (5,5), 0)[..., None]
                    comp = comp*(1 - 0.08*edge_w) + bg_bgr.astype(np.float32)*(0.08*edge_w)
                comp = comp.clip(0,255).astype(np.uint8)
                vis = np.hstack([
                    cv2.resize(pframe, (pframe.shape[1]//2, pframe.shape[0]//2)),
                    cv2.resize(comp, (pframe.shape[1]//2, pframe.shape[0]//2))
                ])
            rows.append(vis)

        grid = np.vstack(rows)
        outdir = self.out_dir.text().strip() or "masks"
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "_preview_compose.jpg")
        cv2.imwrite(path, grid)
        bg.release()
        self._log(f"Предпросмотр сохранён → {path}")

    def _log(self, msg: str):
        self.log.setText(msg); print(msg)

def main():
    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
