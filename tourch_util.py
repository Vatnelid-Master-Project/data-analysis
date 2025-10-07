import numpy as np
import torch
from matplotlib import pyplot as plt
from fastai.vision.all import TensorImage

class TorchUtil:
    def __init__(self, all_spectrograms, all_labels, TARGET_CH, TARGET_HW, EPS, FLOOR_DB):
        self.all_spectrograms = all_spectrograms
        self.all_labels = all_labels
        self.TARGET_CH = TARGET_CH
        self.EPS = EPS
        self.FLOOR_DB = FLOOR_DB
        self.TARGET_HW = TARGET_HW

    def get_x(self, i):
        """
        Convert spectrogram to TensorImage format
        """
        # Normalization - turn spectrogram into dB scale
        arr_db = 10 * np.log10(self.all_spectrograms[i])
        arr_db = np.clip(arr_db, self.FLOOR_DB, 0)

        # Scale to [0, 1]
        arr_01 = (arr_db - self.FLOOR_DB) / (-self.FLOOR_DB)

        # If spectrogram is multi-channel, convert to grayscale first
        if len(arr_db.shape) == 3:
                arr_db = np.mean(arr_db, axis=-1)  # average across channels
        
        arr_01 = arr_01[:, ::-1].copy() # flip spectrogram
        
        # Apply colormap to convert grayscale to color
        # Popular colormaps for spectrograms: 'viridis', 'plasma', 'magma', 'inferno', 'jet', 'hot'
        colormap = plt.get_cmap('magma')  # or try 'plasma', 'magma', 'jet'
        arr_colored = colormap(arr_01)  # This returns RGBA (H, W, 4)
        
        # Convert RGBA to RGB (drop alpha channel)
        arr_rgb = arr_colored[:, :, :3]  # (H, W, 3)
        
        # Convert to PyTorch tensor with correct channel ordering
        t = torch.tensor(arr_rgb).float().permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Resize image
        t_resized = torch.nn.functional.interpolate(t, size=self.TARGET_HW, mode='bilinear', align_corners=False)
        
        return TensorImage(t_resized.squeeze(0))  # (3, H, W)

    def get_spectogram(self, i):
        arr = self.all_spectrograms[i]                 # expected (C, F, T)

    # Ensure 3D
        if arr.ndim == 2:                         # (F, T) -> (1, F, T)
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected (C,F,T), got {arr.shape} at index {i}")

        # Enforce consistent channel count to avoid 4 vs 3 errors
        C = arr.shape[0]
        if self.TARGET_CH is not None:
            if C < self.TARGET_CH:                     # pad/repeat to reach TARGET_CH
                reps = (self.TARGET_CH + C - 1) // C
                arr = np.tile(arr, (reps, 1, 1))[:self.TARGET_CH]
            elif C > self.TARGET_CH:                   # drop extra channels (keep last ones)
                arr = arr[-self.TARGET_CH:, ...]

        # --- Normalization to dB and [0,1] ---
        arr = arr.astype(np.float32)
        arr_db = 10.0 * np.log10(np.maximum(arr, self.EPS))  # guard against <=0
        arr_db = np.clip(arr_db, self.FLOOR_DB, 0.0)
        arr_01 = (arr_db - self.FLOOR_DB) / (-self.FLOOR_DB)      # [0,1]

        # Flip spectrogram (same as your original)
        arr_01 = arr_01[:, ::-1, :].copy()

        # To tensor; keep size unless RESIZE_HW is set
        t = torch.from_numpy(arr_01).float().unsqueeze(0)  # (1, C, F, T)
        if self.TARGET_HW is not None:
            t = torch.nn.functional.interpolate(t, size=self.TARGET_HW, mode='bilinear', align_corners=False)

        return TensorImage(t.squeeze(0).contiguous())      # (C, F, T) or (C, H, W) if resized
    
    def get_labels(self, i):
        return self.all_labels[i]