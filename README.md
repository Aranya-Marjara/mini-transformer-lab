# 🌿 VanaSight: Forest Vision

*From Pixels to Perception — A Complete Computer Vision Pipeline*

> 🧠 **Developed and Tested on Gentoo Linux**
>
> This project was fully built and tested on Gentoo Linux — though it should work on most systems with Python 3.8+, the setup and dependencies are verified only on Linux environments.

---

## 🚀 Installation

### 🧪 Gentoo Linux (Developed & Tested Platform)
```bash
git clone https://github.com/Aranya-Marjara/VanaSight.git
cd VanaSight
pip install -e .
```

### 🐧 Ubuntu / Debian
```bash
sudo apt update
sudo apt install python3-pip python3-venv -y
git clone https://github.com/Aranya-Marjara/VanaSight.git
cd VanaSight
pip install -e .
```

### 🪟 Windows
```bash
# Install Python 3.8+ from python.org first
git clone https://github.com/Aranya-Marjara/VanaSight.git
cd VanaSight
pip install -e .
```

### 🍎 macOS
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python
git clone https://github.com/Aranya-Marjara/VanaSight.git
cd VanaSight
pip install -e .
```

---

## ⚡ Usage

```bash
# Basic usage
vanasight --input image.jpg --output results.jpg

# From URL
vanasight --input "https://picsum.photos/800/600" --output output.jpg

# Force CPU mode
vanasight --input image.jpg --device cpu
```

---

## Real-World Applications

> **Educational Focus**
> 
> VanaSight is built for educational purposes to understand how computer vision systems work under the hood. It's designed for learning, not for competing with production systems.

| Stage | Real-World Use Case | Techniques |
|-------|---------------------|------------|
| Image Loading | CCTV, Medical scans, Satellite data | OpenCV, Requests |
| Noise Removal | Surveillance cleanup, Restoration | FastNlMeansDenoising |
| Contrast Enhancement | Satellite, Document recovery | CLAHE Algorithm |
| Sharpening | Forensics, OCR | Custom Convolution Filters |
| AI Classification | Wildlife tracking, Security | ResNet-18 |
| Object Detection | Inventory, Robotics vision | Canny + Contour filters |
| AI Art Generator | Marketing, Stylized visuals | PIL Filters + Color Mapping |

### Example Use Cases

- **Security**: Enhance CCTV footage, detect objects, classify activities
- **E-Commerce**: Enhance product photos, detect boundaries, create variants
- **Medical Education**: Demonstrate image enhancement concepts
- **Education**: Teach complete computer vision pipeline visually
- **Creative**: Automatically stylize images for content creation

---

## 🧩 How It Works — Flowchart

```text
Input Image
    │
    ▼
[Load Image Module]
 ├── Check local file
 ├── If fails → try URL
 └── If fails → generate synthetic image
    │
    ▼
[Enhancement Pipeline]
 ├── Denoise → CLAHE → Sharpen
 ├── Boost colors → Gamma correct
    │
    ▼
[AI Classification]
 ├── Preprocess (resize + normalize)
 ├── ResNet-18 inference
 └── Return top-3 predictions
    │
    ▼
[Object Detection]
 ├── Convert grayscale → Canny
 ├── Contour filter + labeling
 └── Draw bounding boxes
    │
    ▼
[AI Art Generator]
 ├── Analyze class → Apply style:
 │     ├─ Nature → Dreamy
 │     ├─ Portraits → Warm focus
 │     └─ Others → High contrast
 └── Add vignette + texture
    │
    ▼
[Visualization]
 └── Combine all in 2×2 grid → Save result.jpg
```

---

## ⚙️ Technical Stack

| Component | Library / Framework |
| ---------- | ------------------ |
|  AI Model | PyTorch (ResNet-18) |
|  Image I/O | OpenCV, PIL |
|  CLI | argparse |
|  Packaging | setuptools |
|  Denoising | OpenCV FastNlMeans |
|  Enhancement | CLAHE, Gamma |
|  Detection | Canny + Contours |
|  Style Transfer | Custom filters |

---

## Developer Notes

- Built and tested **exclusively on Gentoo Linux**
- Debugging was done with the help of AI (You should not expect more from a self-taught Python programmer)
- GPU support optional (PyTorch auto-detects CUDA)
- Modular pipeline: every stage can run independently
- Ideal for demos, CV research prototypes, or AI art workflows
---

## 🐾 Aranya-Marjara Collective

> “Where Code Meets the Wilderness.”

A non-profit, open-source collective building ethical AI ecosystems —  
balancing the grace of the forest and the wisdom of the wild.

🌐 [https://github.com/Aranya-Marjara](https://github.com/Aranya-Marjara)

---
