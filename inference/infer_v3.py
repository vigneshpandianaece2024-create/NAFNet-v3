import os
import zipfile
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# ==============================================================================
# 1. EMBEDDED NAFNET v3 ARCHITECTURE 
# ==============================================================================
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b

class LocalAvgPool2d(nn.Module):
    def __init__(self, base_size=None):
        super().__init__()
        self.base_size = base_size

    def forward(self, x):
        if self.base_size is None:
            return F.adaptive_avg_pool2d(x, 1)
        h, w = x.shape[-2:]
        kh, kw = min(self.base_size, h), min(self.base_size, w)
        if kh >= h and kw >= w:
            return F.adaptive_avg_pool2d(x, 1)
        s = torch.cumsum(torch.cumsum(x, dim=-1), dim=-2)
        s = F.pad(s, (1, 0, 1, 0))
        out = (s[..., kh:, kw:] + s[..., :-kh, :-kw]
               - s[..., :-kh, kw:] - s[..., kh:, :-kw]) / (kh * kw)
        return F.pad(out, (kw // 2, (kw - 1) // 2, kh // 2, (kh - 1) // 2),
                     mode='replicate')

class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_c, ffn_c = c * dw_expand, c * ffn_expand

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c)
        self.sg = SimpleGate()
        self.pool = LocalAvgPool2d()
        self.sca_conv = nn.Conv2d(dw_c // 2, dw_c // 2, 1)
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1)

        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_c, 1)
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1)

        self.drop1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca_conv(self.pool(x))
        x = self.conv3(x)
        y = inp + self.drop1(x) * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.drop2(x) * self.gamma

class NAFNetBody(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, width=32,
                 middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8),
                 dec_blk_nums=(2, 2, 2, 2), drop_out_rate=0.0):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1)

        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()

        chan = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(
                *[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)])

        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(
                *[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(n)]))

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x

    def forward(self, inp):
        _, _, H, W = inp.shape
        x_in = self.check_image_size(inp)
        x = self.intro(x_in)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)

        return self.ending(x)[:, :, :H, :W]

class NAFNetSR(nn.Module):
    def __init__(self, sf=2, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2),
                 in_channels=1, drop_out_rate=0.0):
        super().__init__()
        self.sf = sf
        self.body = NAFNetBody(in_channels=in_channels, out_channels=width,
                               width=width, middle_blk_num=middle_blk_num,
                               enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums,
                               drop_out_rate=drop_out_rate)
        if sf > 1:
            self.head = nn.Sequential(
                nn.Conv2d(width, width * 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width * 2, width * sf * sf, 3, padding=1),
                nn.PixelShuffle(sf),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, 1, 3, padding=1))
        else:
            self.head = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, x):
        base = (F.interpolate(x, scale_factor=self.sf, mode='bilinear',
                              align_corners=False) if self.sf > 1 else x)
        return self.head(self.body(x)) + base

def enable_tlc(model, train_patch):
    n = 0
    for m in model.modules():
        if isinstance(m, LocalAvgPool2d):
            m.base_size = train_patch
            n += 1
    for depth, stage in enumerate(getattr(model.body, 'encoders', [])):
        for blk in stage.modules():
            if isinstance(blk, LocalAvgPool2d):
                blk.base_size = max(4, train_patch // (2 ** depth))
    return n

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def to_hw(a):
    a = np.asarray(a)
    if a.ndim == 2: return a
    if a.ndim == 3:
        if a.shape[0] == 1: return a[0]
        if a.shape[2] == 1: return a[:, :, 0]
    raise ValueError(f'Unsupported shape {a.shape}')

def load_weights_from_zip(zip_path, device):
    """Extracts the zip, finds the model weights, and repackages if needed."""
    print(f"Loading weights from {zip_path}...")
    temp_dir = tempfile.mkdtemp()
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            if f.endswith('.pth'):
                pth_path = os.path.join(root, f)
                checkpoint = torch.load(pth_path, map_location=device)
                shutil.rmtree(temp_dir)
                return checkpoint
    
    data_pkl_dir = None
    for root, dirs, files in os.walk(temp_dir):
        if 'data.pkl' in files:
            data_pkl_dir = root
            break
            
    if data_pkl_dir:
        repacked_pth = 'temp_repacked.pth'
        with zipfile.ZipFile(repacked_pth, 'w', zipfile.ZIP_STORED) as zipf:
            for r, _, f in os.walk(data_pkl_dir):
                for file in f:
                    file_path = os.path.join(r, file)
                    rel_path = os.path.relpath(file_path, data_pkl_dir)
                    arcname = os.path.join('archive', rel_path)
                    zipf.write(file_path, arcname)
        
        checkpoint = torch.load(repacked_pth, map_location=device)
        os.remove(repacked_pth)
        shutil.rmtree(temp_dir)
        return checkpoint
        
    shutil.rmtree(temp_dir)
    raise FileNotFoundError("Could not find a valid .pth file in the zip.")

# ==============================================================================
# 3. INTERACTIVE INFERENCE PIPELINE
# ==============================================================================

# ==============================================================================
# 3. INTERACTIVE INFERENCE PIPELINE
# ==============================================================================

def main():
    # --- MODEL CONFIGURATION ---
    zip_path = 'best.zip'   
    width = 32
    train_patch = 96
    sf = 2
    # ---------------------------
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Initializing environment...")
    
    # 1. LOAD CHECKPOINT
    if not os.path.exists(zip_path):
        print(f"Error: Could not find '{zip_path}'. Make sure it's in the same folder as this script.")
        return

    checkpoint = load_weights_from_zip(zip_path, device)
    
    model = NAFNetSR(sf=sf, width=width).to(device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        scale = checkpoint.get('scale', 255.0)
    else:
        model.load_state_dict(checkpoint)
        scale = 255.0

    model.eval()
    enable_tlc(model, train_patch)
    print("\n" + "="*50)
    print(f"✅ Model successfully loaded on {device.upper()} and ready for inference!")
    print("="*50 + "\n")

    # 2. INTERACTIVE LOOP
    while True:
        print("Type 'q' or 'quit' to exit.")
        raw_input = input("Drag and drop an image file here (or paste the path): ").strip()
        
        if raw_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting interactive mode. Goodbye!")
            break
            
        # Remove quotes that terminals often add when drag-and-dropping files
        img_path = raw_input.strip("\"'") 
        p = Path(img_path)
        
        if not p.exists() or not p.is_file():
            print(f"❌ Error: Could not find file at '{img_path}'. Please try again.\n")
            continue
            
        if p.suffix.lower() not in ['.npy', '.png', '.jpg', '.jpeg']:
            print(f"❌ Error: Unsupported file type ({p.suffix}). Please use .npy, .png, or .jpg.\n")
            continue

        print(f"Processing '{p.name}'...")

        try:
            with torch.no_grad():
                # Load Data Based on File Type
                if p.suffix.lower() == '.npy':
                    lr_img = to_hw(np.load(p)).astype(np.float32) / scale
                else:
                    img = Image.open(p).convert('L') # Convert to Grayscale
                    lr_img = np.array(img).astype(np.float32) / 255.0
                
                # Convert to Tensor [1, 1, H, W]
                lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_img)).unsqueeze(0).unsqueeze(0).to(device)
                
                # Forward Pass
                out_tensor = model(lr_tensor)
                
                # Convert back to Numpy for visualization
                out_img = out_tensor.squeeze().cpu().numpy()
                out_img_scaled = np.clip(out_img, 0, 1)
                
                # Original image scaled for visualization
                in_img_scaled = np.clip(lr_img, 0, 1)

            # EXTRACT REAL DIMENSIONS (Height, Width)
            in_h, in_w = in_img_scaled.shape
            out_h, out_w = out_img_scaled.shape

            # 3. DISPLAY THE INTERACTIVE PLOT
            plt.style.use('dark_background') # Looks great for demo videos
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.canvas.manager.set_window_title(f"NAFNet Denoising Demo - {p.name}")
            
            # Subplot 1: Noisy Input with explicit dimensions
            axes[0].imshow(in_img_scaled, cmap='gray')
            axes[0].set_title(f"Noisy Input\n[{in_w} x {in_h} pixels]", fontsize=16, fontweight='bold', pad=15)
            axes[0].axis('off')
            
            # Subplot 2: Restored Output with explicit dimensions
            axes[1].imshow(out_img_scaled, cmap='gray')
            axes[1].set_title(f"Restored & Upsampled Output\n[{out_w} x {out_h} pixels]", fontsize=16, fontweight='bold', pad=15, color='#00ffcc') # Added a slight color pop to emphasize the output
            axes[1].axis('off')
            
            plt.tight_layout()
            
            print("Showing result window... (Close the window to process the next image)")
            plt.show() # This pauses the terminal loop until you close the image window
            print("\nReady for next image.")
            
        except Exception as e:
            print(f"❌ An error occurred during processing: {e}\n")

if __name__ == '__main__':
    main()
