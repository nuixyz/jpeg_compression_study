import cv2
import io
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fftpack import dctn, idctn
from skimage.metrics import structural_similarity as ssim

# ─── Quantization Matrix (standard JPEG luminance) ───────────────────────────
BASE_QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


def get_quantization_matrix(quality=50):
    """Scale quantization matrix by JPEG quality factor (1–100)."""
    quality = np.clip(quality, 1, 100)
    scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
    return np.clip((BASE_QUANTIZATION_MATRIX * scale + 50) / 100, 1, 255)


def process_image_dct(image, quant_matrix):
    """Vectorized DCT compression — processes all 8x8 blocks at once."""
    h, w = image.shape
    img_float = np.float32(image) - 128

    # Reshape to (h//8, w//8, 8, 8) block grid
    blocks = img_float.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)

    # Forward DCT → quantize → dequantize → inverse DCT
    dct_blocks = dctn(blocks, axes=(-2, -1), norm='ortho')
    quantized = np.round(dct_blocks / quant_matrix)
    dequantized = quantized * quant_matrix
    recon = idctn(dequantized, axes=(-2, -1), norm='ortho')

    # Reshape back and clip
    result = recon.transpose(0, 2, 1, 3).reshape(h, w) + 128
    return np.clip(result, 0, 255).astype(np.uint8)


def compute_metrics(original, compressed):
    """Return MSE, PSNR, and SSIM."""
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    psnr = 100.0 if mse == 0 else 20 * np.log10(255 / np.sqrt(mse))
    ssim_score = ssim(original, compressed)
    return mse, psnr, ssim_score


def get_image_size_kb(img_array, quality=95):
    """Encode a numpy array as JPEG in-memory and return its size in KB."""
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    return buf.tell() / 1024  # bytes → KB


def visualize_dct_coefficients(image):
    """Show DCT coefficient energy map of the full image."""
    img_float = np.float32(image) - 128
    blocks = img_float.reshape(image.shape[0] // 8, 8, image.shape[1] // 8, 8).transpose(0, 2, 1, 3)
    dct_blocks = dctn(blocks, axes=(-2, -1), norm='ortho')
    # Average magnitude across all blocks
    avg_coeff = np.mean(np.abs(dct_blocks), axis=(0, 1))

    plt.figure(figsize=(5, 4))
    plt.imshow(np.log(avg_coeff + 1), cmap='hot')
    plt.colorbar(label='log(|DCT coeff| + 1)')
    plt.title("Average DCT Coefficient Magnitude\n(log scale — energy concentrates top-left)")
    plt.xlabel("Horizontal frequency")
    plt.ylabel("Vertical frequency")
    plt.tight_layout()
    plt.savefig("dct_coefficients.png", dpi=150)
    plt.show()


# ─── Load & Preprocess Image ─────────────────────────────────────────────────
image = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image not found. Place 'sample.jpg' inside an 'images/' folder.")
    exit()

# Crop to nearest multiple of 8
h, w = image.shape
image = image[:h - h % 8, :w - w % 8]

# Original file size from disk
original_disk_kb = os.path.getsize("images/sample.jpg") / 1024
original_encoded_kb = get_image_size_kb(image)
print(f"Image dimensions : {image.shape}")
print(f"Original size    : {original_disk_kb:.2f} KB (disk) | {original_encoded_kb:.2f} KB (re-encoded JPEG)")
print("-" * 70)
print(f"{'Q':>4} | {'File Size (KB)':>14} | {'Ratio':>7} | {'MSE':>8} | {'PSNR (dB)':>9} | {'SSIM':>6}")
print("-" * 70)

# ─── Multi-Quality Comparison ─────────────────────────────────────────────────
quality_levels = [5, 25, 50, 75, 95]
results = []

for q in quality_levels:
    qm = get_quantization_matrix(q)
    comp = process_image_dct(image, qm)
    mse, psnr, ssim_score = compute_metrics(image, comp)
    comp_kb = get_image_size_kb(comp, quality=q)
    ratio = original_disk_kb / comp_kb if comp_kb > 0 else float('inf')
    results.append((q, comp, mse, psnr, ssim_score, comp_kb, ratio))
    print(f"Q={q:3d} | {comp_kb:>13.2f} | {ratio:>6.2f}x | {mse:>8.2f} | {psnr:>9.2f} | {ssim_score:.4f}")

print("-" * 70)

# ─── Plot: Original + Compressed at Each Quality ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

axes[0].imshow(image, cmap='gray')
axes[0].set_title(f"Original Image\n{original_disk_kb:.1f} KB (disk)", fontsize=13, fontweight='bold')
axes[0].axis("off")

for idx, (q, comp, mse, psnr, ssim_score, comp_kb, ratio) in enumerate(results, start=1):
    axes[idx].imshow(comp, cmap='gray')
    axes[idx].set_title(
        f"Quality = {q} | {comp_kb:.1f} KB ({ratio:.1f}x smaller)\nPSNR={psnr:.1f} dB | SSIM={ssim_score:.3f}",
        fontsize=9
    )
    axes[idx].axis("off")

plt.suptitle("JPEG DCT Compression at Multiple Quality Levels", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("compression_comparison.png", dpi=150)
plt.show()

# ─── Plot: Metrics vs Quality ─────────────────────────────────────────────────
qs    = [r[0] for r in results]
psnrs = [r[3] for r in results]
ssims = [r[4] for r in results]
sizes = [r[5] for r in results]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.plot(qs, psnrs, 'bo-', linewidth=2, markersize=7)
ax1.set_xlabel("Quality Factor"); ax1.set_ylabel("PSNR (dB)")
ax1.set_title("PSNR vs Quality"); ax1.grid(True, alpha=0.4)

ax2.plot(qs, ssims, 'rs-', linewidth=2, markersize=7)
ax2.set_xlabel("Quality Factor"); ax2.set_ylabel("SSIM")
ax2.set_title("SSIM vs Quality"); ax2.grid(True, alpha=0.4)

ax3.plot(qs, sizes, 'g^-', linewidth=2, markersize=7)
ax3.axhline(original_disk_kb, color='gray', linestyle='--', linewidth=1.5, label=f'Original ({original_disk_kb:.1f} KB)')
ax3.set_xlabel("Quality Factor"); ax3.set_ylabel("File Size (KB)")
ax3.set_title("File Size vs Quality"); ax3.legend(); ax3.grid(True, alpha=0.4)

plt.suptitle("Image Quality Metrics & File Size vs Compression Quality Factor", fontsize=13)
plt.tight_layout()
plt.savefig("metrics_vs_quality.png", dpi=150)
plt.show()

# ─── DCT Coefficient Visualization ───────────────────────────────────────────
visualize_dct_coefficients(image)

print("\nDone! Saved: compression_comparison.png, metrics_vs_quality.png, dct_coefficients.png")