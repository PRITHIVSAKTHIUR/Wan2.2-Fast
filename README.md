# **[Wan2.2-Fast](https://huggingface.co/spaces/prithivMLmods/Wan2.2-Fast)**

Wan2.2-Fast is an optimized, high-performance image-to-video (I2V) generation suite powered by the `Wan-AI/Wan2.2-I2V-A14B-Diffusers` model. Utilizing dual transformer backbones (`cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers`), fused step-distilled Lightning LoRA adapters (`Kijai/WanVideo_comfy`), and torchao-driven FP8/Int8 quantization, the pipeline enables rapid 4-step video synthesis directly on CUDA hardware.

The application integrates ahead-of-time (AOTI) compilation modules (`cbensimon/WanTransformer3DModel-sm120-cu130-raa`) on ZeroGPU infrastructure and serves an interactive web experience through a custom dark-mode single-page frontend (SPA) managed by FastAPI and `gradio.Server`.

https://github.com/user-attachments/assets/9a867eff-b32f-4a2d-97f4-94e4213c5477

### **Key Features**

* **Wan 2.2 14B Dual-Transformer Architecture:** Leverages dual 3D transformer modules (`transformer` and `transformer_2`) in BF16 precision for temporal video rendering.
* **Fused 4-Step Lightning Distillation:** Fuses `Lightx2v` LoRA adapters at scaled components ($3.0\times$ on `transformer`, $1.0\times$ on `transformer_2`) to achieve high-fidelity video generation in 4–6 inference steps.
* **FP8 & Int8 Quantization via `torchao`:** Quantizes the primary text encoder with `Int8WeightOnlyConfig` and transformer backbones with `Float8DynamicActivationFloat8WeightConfig` for reduced VRAM consumption.
* **AOTI Compilation Integration:** Implements `aoti.py` hooks to load pre-compiled AOTInductor packages (`package.pt2`) across ZeroGPU environments.
* **Studio SPA Interface:** An interactive single-page application built with modern vanilla web components—featuring video loop playback, history filmstrips, prompt quick-actions, and drag-and-drop file uploaders.
* **Smart Boundary Resizing:** Dynamically resizes input imagery within a 480px–832px window, snapping dimensions to multiples of 16 to preserve aspect ratios and prevent tensor shape mismatches.

### **Repository Structure**

```text
├── example-file/
│   ├── 6b2842cf438d086f556eef05cc29d2d1.jpg
│   ├── kill_bill.jpeg
│   ├── wan_i2v_input.JPG
│   └── wan22_input_2.jpg
├── aoti.py
├── app.py
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the Wan2.2-Fast environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Python **3.12 or above** is required and recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for optimal stability and quantization support.
* **CUDA Version:** **CUDA 13.0** is recommended (`--extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)`), matching the live Hugging Face Space configuration.


#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment synchronization and deterministic dependency management based on `uv.lock`.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Wan2.2-Fast.git
cd Wan2.2-Fast
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the script**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install pip>=26.1.2
```

**2. Install Core Dependencies**
Install the deep learning stack, diffusion libraries, quantization framework, and video utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130

transformers==5.15.0
accelerate==1.14.0
diffusers==0.39.0
spaces==0.51.1
peft==0.20.0
gradio==6.22.0
pillow==12.3.0
huggingface-hub>=1.27.0
opencv-python==5.0.0.93
torch==2.11.0
torchvision==0.26.0
torchao==0.18.0
sentencepiece==0.2.2
safetensors==0.8.0
imageio==2.37.4
imageio-ffmpeg==0.6.0
ftfy==6.3.1
```

### **Usage**

Once the FastAPI web server initializes, navigate to your local loopback address (typically `http://127.0.0.1:7860/`).

1. **Upload Input Image:** Drag and drop an image into the central canvas, paste from clipboard, or click the upload icon in the left rail.
2. **Configure Video Parameters:**
* **Instruction:** Enter custom motion prompts (e.g., *"POV selfie video, white cat on surfboard"*).
* **Duration:** Adjust the length slider between 0.5s and 5.0s (rendered at 16 FPS).
* **Steps & CFG:** Configure inference steps (default 4) and dual guidance scale values (`CFG 1` and `CFG 2`).


3. **Generate:** Press ⌘/Ctrl + Enter or click **Generate Video** to start the CUDA inference queue.
4. **Playback & Export:** Preview the generated MP4 video directly on the looped canvas player, or click the download icon to save the file locally.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Wan2.2-Fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Wan2.2-Fast.git](https://github.com/PRITHIVSAKTHIUR/Wan2.2-Fast.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Wan2.2-Fast](https://huggingface.co/spaces/prithivMLmods/Wan2.2-Fast)
