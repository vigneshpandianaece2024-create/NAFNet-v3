"""
NAFNet v3 - matched to the official SIDD denoising configuration.

Block layout follows megvii-research/NAFNet, options/train/SIDD/NAFNet-width32.yml:

    enc_blk_nums   [2, 2, 4, 8]
    middle_blk_num 12
    dec_blk_nums   [2, 2, 2, 2]

The GoPro deblurring configuration is different:
    [1, 1, 1, 28] with 1 middle block.

Denoising is the closer analogue to this task, so the SIDD shape is used.

Beyond the reference network:
  - a pixel-shuffle SR head, since the task is 2x super-resolution as well
    as restoration
  - TLC local pooling, switchable at inference

Reference:
Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

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
    """
    Global average pooling during training; windowed at inference when TLC
    is enabled.

    NAFNet's channel attention pools over the whole feature map. A model
    trained on 96px crops but run on 128px images sees different pooled
    statistics than its weights expect, and quality drops. Capping the
    window to the training size removes that mismatch.

    Uses a summed-area table, so cost is independent of window size.
    """

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

        out = (
            s[..., kh:, kw:]
            + s[..., :-kh, :-kw]
            - s[..., :-kh, kw:]
            - s[..., kh:, :-kw]
        ) / (kh * kw)

        return F.pad(
            out,
            (
                kw // 2,
                (kw - 1) // 2,
                kh // 2,
                (kh - 1) // 2,
            ),
            mode="replicate",
        )


# ---------------------------------------------------------------------------
# NAF Block
# ---------------------------------------------------------------------------

class NAFBlock(nn.Module):
    def __init__(
        self,
        c,
        dw_expand=2,
        ffn_expand=2,
        drop_out_rate=0.0,
    ):
        super().__init__()

        dw_c = c * dw_expand
        ffn_c = c * ffn_expand

        self.norm1 = LayerNorm2d(c)

        self.conv1 = nn.Conv2d(c, dw_c, 1)

        self.conv2 = nn.Conv2d(
            dw_c,
            dw_c,
            3,
            padding=1,
            groups=dw_c,
        )

        self.sg = SimpleGate()

        self.pool = LocalAvgPool2d()

        self.sca_conv = nn.Conv2d(
            dw_c // 2,
            dw_c // 2,
            1,
        )

        self.conv3 = nn.Conv2d(
            dw_c // 2,
            c,
            1,
        )

        self.norm2 = LayerNorm2d(c)

        self.conv4 = nn.Conv2d(
            c,
            ffn_c,
            1,
        )

        self.conv5 = nn.Conv2d(
            ffn_c // 2,
            c,
            1,
        )

        self.drop1 = (
            nn.Dropout(drop_out_rate)
            if drop_out_rate > 0
            else nn.Identity()
        )

        self.drop2 = (
            nn.Dropout(drop_out_rate)
            if drop_out_rate > 0
            else nn.Identity()
        )

        # Zero initialization makes every block start close to identity,
        # which helps stabilize optimization of deep stacks.
        self.beta = nn.Parameter(
            torch.zeros(1, c, 1, 1)
        )

        self.gamma = nn.Parameter(
            torch.zeros(1, c, 1, 1)
        )

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


# ---------------------------------------------------------------------------
# NAFNet Body
# ---------------------------------------------------------------------------

class NAFNetBody(nn.Module):
    """
    U-shaped NAFNet.

    Returns feature maps at input resolution.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
        drop_out_rate=0.0,
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels,
            width,
            3,
            padding=1,
        )

        self.ending = nn.Conv2d(
            width,
            out_channels,
            3,
            padding=1,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width

        # Encoder
        for n in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            chan,
                            drop_out_rate=drop_out_rate,
                        )
                        for _ in range(n)
                    ]
                )
            )

            self.downs.append(
                nn.Conv2d(
                    chan,
                    2 * chan,
                    2,
                    stride=2,
                )
            )

            chan *= 2

        # Middle blocks
        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(
                    chan,
                    drop_out_rate=drop_out_rate,
                )
                for _ in range(middle_blk_num)
            ]
        )

        # Decoder
        for n in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        chan,
                        chan * 2,
                        1,
                        bias=False,
                    ),
                    nn.PixelShuffle(2),
                )
            )

            chan //= 2

            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            chan,
                            drop_out_rate=drop_out_rate,
                        )
                        for _ in range(n)
                    ]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.shape

        ph = (
            self.padder_size - h % self.padder_size
        ) % self.padder_size

        pw = (
            self.padder_size - w % self.padder_size
        ) % self.padder_size

        if ph or pw:
            return F.pad(
                x,
                (0, pw, 0, ph),
                mode="reflect",
            )

        return x

    def forward(self, inp):
        _, _, H, W = inp.shape

        x_in = self.check_image_size(inp)

        x = self.intro(x_in)

        skips = []

        for enc, down in zip(
            self.encoders,
            self.downs,
        ):
            x = enc(x)

            skips.append(x)

            x = down(x)

        x = self.middle_blks(x)

        for dec, up, skip in zip(
            self.decoders,
            self.ups,
            skips[::-1],
        ):
            x = up(x)

            x = x + skip

            x = dec(x)

        return self.ending(x)[:, :, :H, :W]


# ---------------------------------------------------------------------------
# NAFNet SR
# ---------------------------------------------------------------------------

class NAFNetSR(nn.Module):
    """
    Restoration body + pixel-shuffle SR head.

    A bilinear skip carries the low frequencies, so the network only has to
    learn the residual detail rather than reproducing the whole image.
    """

    def __init__(
        self,
        sf=2,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=(2, 2, 4, 8),
        dec_blk_nums=(2, 2, 2, 2),
        in_channels=1,
        drop_out_rate=0.0,
    ):
        super().__init__()

        self.sf = sf

        self.body = NAFNetBody(
            in_channels=in_channels,
            out_channels=width,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
            drop_out_rate=drop_out_rate,
        )

        if sf > 1:
            self.head = nn.Sequential(
                nn.Conv2d(
                    width,
                    width * 2,
                    3,
                    padding=1,
                ),

                nn.GELU(),

                nn.Conv2d(
                    width * 2,
                    width * sf * sf,
                    3,
                    padding=1,
                ),

                nn.PixelShuffle(sf),

                nn.Conv2d(
                    width,
                    width,
                    3,
                    padding=1,
                ),

                nn.GELU(),

                nn.Conv2d(
                    width,
                    1,
                    3,
                    padding=1,
                ),
            )

        else:
            self.head = nn.Sequential(
                nn.Conv2d(
                    width,
                    width,
                    3,
                    padding=1,
                ),

                nn.GELU(),

                nn.Conv2d(
                    width,
                    1,
                    3,
                    padding=1,
                ),
            )

    def forward(self, x):
        if self.sf > 1:
            base = F.interpolate(
                x,
                scale_factor=self.sf,
                mode="bilinear",
                align_corners=False,
            )
        else:
            base = x

        return self.head(self.body(x)) + base


# ---------------------------------------------------------------------------
# TLC
# ---------------------------------------------------------------------------

def enable_tlc(model, train_patch):
    """
    Enable TLC after loading weights and before inference.

    train_patch:
        LR training patch size.
    """

    n = 0

    for m in model.modules():
        if isinstance(m, LocalAvgPool2d):
            m.base_size = train_patch
            n += 1

    for depth, stage in enumerate(
        getattr(model.body, "encoders", [])
    ):
        for blk in stage.modules():
            if isinstance(blk, LocalAvgPool2d):
                blk.base_size = max(
                    4,
                    train_patch // (2 ** depth),
                )

    return n


def disable_tlc(model):
    """
    Disable TLC and restore global average pooling.
    """

    for m in model.modules():
        if isinstance(m, LocalAvgPool2d):
            m.base_size = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    torch.manual_seed(0)

    # Parameter count for several model widths
    for w in (16, 24, 32, 48):
        m = NAFNetSR(
            sf=2,
            width=w,
        )

        print(
            f"width={w:3d}  "
            f"params="
            f"{sum(p.numel() for p in m.parameters()) / 1e6:7.2f}M"
        )

    # Default NAFNet v3 model
    model = NAFNetSR(
        sf=2,
        width=32,
    )

    # Shape tests
    for shape in [
        (2, 1, 96, 96),
        (2, 1, 64, 64),
        (1, 1, 128, 128),
        (1, 1, 67, 93),
    ]:

        y = model(
            torch.randn(*shape)
        )

        exp = (
            shape[0],
            1,
            shape[2] * 2,
            shape[3] * 2,
        )

        assert y.shape == exp, (
            f"{shape} -> {y.shape}, "
            f"expected {exp}"
        )

        print(
            f"{shape} -> {tuple(y.shape)}"
        )

    # TLC test
    n = enable_tlc(
        model,
        96,
    )

    print(
        f"\nTLC enabled on {n} blocks"
    )

    print(
        f"full image out: "
        f"{tuple(model(torch.randn(1, 1, 128, 128)).shape)}"
    )

    # Gradient test
    model(
        torch.randn(1, 1, 64, 64)
    ).sum().backward()

    ng = sum(
        1
        for p in model.parameters()
        if p.grad is not None
    )

    total_params = sum(
        1
        for _ in model.parameters()
    )

    print(
        f"params receiving grad: "
        f"{ng}/{total_params}"
    )

    print("\nOK")
