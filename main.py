import cv2
import io
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dctn, idctn
from skimage.metrics import structural_similarity as ssim

BASE_QUANTIZATION_MATRIX = np.array([
  [16,11,10,16,24,40,51,61],
  [12,12,14,19,26,58,60,55],
  [14,13,16,24,40,57,69,56],
  [14,17,22,29,51,87,80,62],
  [18,22,37,56,68,109,103,77],
  [24,35,55,64,81,104,113,92],
  [49,64,78,87,103,121,120,101],
  [72,92,95,98,112,100,103,99]
], dtype=np.float32)

def get_quantization_matrix(quality=50):
  quality = np.clip(quality, 1, 100)
  scale = (5000 / quality) if quality < 50 else (200 - 2 * quality)
  return np.clip((BASE_QUANTIZATION_MATRIX * scale + 50) / 100, 1, 255)

def process_image_dct(image, quantization_matrix):
  h, w = image.shape
  img_float = np.float32(image) - 128

  blocks = img_float.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)

  dct_blocks = dctn(blocks, axes=(-2, -1), norm='ortho')
  quantized = np.round(dct_blocks / quantization_matrix)
  dequantized = quantized * quantization_matrix
  reconstruct = idctn(dequantized, axes=(-2, -1), norm='ortho')

  result = reconstruct.transpose(0, 2, 1, 3).reshape(h, w) + 128

  return np.clip(result, 0, 255).astype(np.uint8)

def visualize_dct_coefficients(image):
  img_float = np.float32(image) - 128
  blocks = img_float.reshape(image.shape[0] // 8, 8, image.shape[1] // 8, 8).transpose(0, 2, 1, 3)
  dct_blocks = dctn(blocks, axes=(-2, -1), norm='ortho')

  avg_coeff = np.mean(np.abs(dct_blocks), axis=(0,1))

  plt.figure(figsize=(5, 4))
  plt.imshow(np.log(avg_coeff + 1), cmap='hot')
  plt.colorbar(label='log(|DCT coeff| + 1)')
  plt.title("Average DCT Coefficient Magnitude\n(log scale — energy concentrates top-left)")
  plt.xlabel("Horizontal frequency")
  plt.ylabel("Vertical frequency")
  plt.tight_layout()
  plt.savefig("dct_coefficients.png", dpi=150)
  plt.show()

def process_ycbcr(image_ycbcr, quality=50):
    qm = get_quantization_matrix(quality)
    
    chroma_qm = np.clip(qm * 2, 1, 255)
    
    channels = cv2.split(image_ycbcr)
    processed = [
        process_image_dct(channels[0], qm),
        process_image_dct(channels[1], chroma_qm),
        process_image_dct(channels[2], chroma_qm),
    ]
    return cv2.merge(processed)

def compute_metrics(original, compressed):
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    psnr_db = min(20 * np.log10(255 / np.sqrt(mse)), 48.1)
    
    psnr_percent = (psnr_db / 48.5) * 100
    ssim_score = ssim(original, compressed)

    return mse, psnr_percent, ssim_score

def compute_size(img_ycrcb, quality=95):
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    return buf.tell() / 1024

image = cv2.imread("images/cat.png", cv2.IMREAD_COLOR)
if image is None:
    print("Image not found.")
    exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

h, w = image.shape[:2]
image = image[:h - h % 8, :w - w % 8]
print(f"Image size: {image.shape}")

original_size = os.path.getsize("images/cat.png") / 1024
original_encoded = compute_size(image)
print(f"Image dimensions: {image.shape}")
print(f"Original Size: {original_size: .2f} KB")
print(f"Original Encoded Size: {original_encoded: .2f} KB")

quality_levels = [5, 25, 50, 75, 95]
results = []

for q in quality_levels:
  comp = process_ycbcr(image, q) 

  comp_bgr = cv2.cvtColor(comp, cv2.COLOR_YCrCb2BGR)
  orig_bgr = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

  mse, psnr_percent, ssim_score = compute_metrics(
  cv2.split(image)[0],
  cv2.split(comp)[0]
  )

  mse, psnr_percent, ssim_score = compute_metrics(cv2.split(image)[0], cv2.split(comp)[0])
  comp_kb = compute_size(comp, quality=q)
  ratio = original_size / comp_kb
  results.append((q, comp, mse, psnr_percent, ssim_score, comp_kb, ratio))

  print("\n")
  print(f"Compressed size in KB: {comp_kb:3.2f}")
  print(f"Quality Level: {q:3d}")
  print(f"MSE: {mse:8.2f}")
  print(f"PSNR: {psnr_percent:6.2f}")
  print(f"SSIM: {ssim_score:4f}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

orig_bgr = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
axes[0].imshow(orig_bgr[:,:,::-1])
axes[0].set_title(f"Original Image\n{original_size:.1f} KB", fontsize=13, fontweight='bold')
axes[0].axis("off")

for idx, (q, comp, mse, psnr_percent, ssim_score, comp_kb, ratio) in enumerate(results, start=1):
  comp_rgb = cv2.cvtColor(comp, cv2.COLOR_YCrCb2BGR)[:,:,::-1]
  axes[idx].imshow(comp_rgb)
  axes[idx].set_title(
    f"Quality = {q} | Compressed size in KB: {comp_kb:.1f} KB \nCompressed Ratio: {ratio:.1f}\nPSNR={psnr_percent:.1f} % | SSIM={ssim_score:.3f}",
    fontsize=10
  )
  axes[idx].axis("off")

plt.suptitle("JPEG DCT Compression at Multiple Quality Levels", fontsize=14, fontweight='bold')
plt.tight_layout()
# plt.savefig("compression_comparison.png", dpi=150)
plt.show()
# print(results)
fig, ax = plt.subplots(figsize=(9, 5))

labels = [f"Q={r[0]}" for r in results]
comp_sizes = [r[5] for r in results]
colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']

bars = ax.bar(labels, comp_sizes, color=colors, edgecolor='black', linewidth=0.8, zorder=3)
ax.axhline(original_size, color='steelblue', linestyle='--', linewidth=2,
           label=f'Original ({original_size:.1f} KB)')

for bar, kb, ratio in zip(bars, comp_sizes, [r[6] for r in results]):
  ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, 
          f"{kb:.1f} KB\n({ratio:.1f}x)", ha='center', va='bottom', fontsize=9, fontweight='bold'
          )
ax.set_xlabel("Quality Factor", fontsize=12)
ax.set_ylabel("File Size (KB)", fontsize=12)
ax.set_title("Compressed Image File Sizes vs Original", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.4, zorder=0)
ax.set_ylim(0, max(original_size, max(comp_sizes)) * 1.3)
plt.tight_layout()
# plt.savefig("size_comparison_bar.png", dpi=150)
plt.show()

qs = [r[0] for r in results]
psnrs = [r[3] for r in results]
ssims = [r[4] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(qs, psnrs, 'bo-', linewidth=2, markersize=7)
ax1.set_xlabel("Quality Factor")
ax1.set_ylabel("PSNR(dB)")
ax1.set_title("PSNR vs Quality")
ax1.grid(True, alpha=0.4)

ax2.plot(qs, ssims, 'rs-', linewidth=2, markersize=7)
ax2.set_xlabel("Quality Factor")
ax2.set_ylabel("SSIM")
ax2.set_title("SSIM vs Quality")
ax2.grid(True, alpha=0.4)
plt.suptitle("Image Quality Metrics vs Compression Quality Factor", fontsize=13)
plt.tight_layout()
# plt.savefig("metrics_vs_quality.png", dpi=150)
plt.show()

visualize_dct_coefficients(cv2.split(image)[0])