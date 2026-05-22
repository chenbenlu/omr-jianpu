# Role and Context
You are a Senior AI/ML DevOps Engineer and Technical Lead. Our 4-person team is developing a Deep Learning End-to-End Optical Music Recognition (OMR) system. The project translates printed sheet music images directly into Jianpu (numbered musical notation) semantic tokens using an Image-to-Sequence (Vision-Encoder-Decoder) architecture.

# Hardware & OS Specs
* **OS:** Windows 11 x64 (Development will be routed through WSL2 for Docker).
* **GPUs:** NVIDIA RTX 5060 Laptop GPU, RTX 5070, and RTX 6000.
* **Compute Requirement:** All three GPUs are Blackwell (compute capability `sm_120`), so we must use **PyTorch 2.7+ built against CUDA 12.8 or newer**. Even PyTorch 2.6 wheels only ship kernels up to `sm_90` (Hopper) and fail at runtime on Blackwell with `no kernel image is available for execution on the device`. Native `sm_120` binaries first appeared in PyTorch 2.7.0 (April 2025).

# Your Task
Please generate the complete foundational repository structure, configuration files, and collaboration guidelines based on the requirements below.

## Part 1: Unified Docker & uv Environment
Provide the exact code for the following files to ensure our environments are strictly identical across all machines:

1. **`Dockerfile`**:
   - Use an official PyTorch base image with PyTorch 2.7+ and CUDA 12.8+ (e.g., `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`). PyTorch 2.6 and earlier (including `2.2.0-cuda12.1`) lack Blackwell `sm_120` kernels and will not run on this team's GPUs even when the CUDA toolkit version looks recent.
   - Install `uv` as the package manager.
   - Set up a non-root user for security and file permission alignment between WSL2 host and container.
   - Configure the entrypoint to keep the container alive for remote development (e.g., VS Code Dev Containers).

2. **`docker-compose.yml`**:
   - Must explicitly include the `deploy.resources.reservations.devices` block to enable GPU passthrough via NVIDIA Container Toolkit for WSL2.
   - Mount the current working directory to `/workspace` inside the container.
   - Map necessary ports (e.g., for JupyterLab, TensorBoard, or Streamlit).

3. **`Makefile`**:
   - Create a Makefile to automate common commands, lowering the barrier for the team. 
   - Include commands like `make build`, `make up`, `make down`, `make shell`, and `make install` (which should run `uv pip install -r requirements.txt` inside the container).

## Part 2: 4-Person GitHub Flow Strategy
Define a strict yet agile Git workflow tailored for our 4 modules:
* Member A: Data Engineering (Camera-PrIMuS parsing, augmentation, Dataloaders).
* Member B: Model Training (Vision-Encoder-Decoder fine-tuning, loss functions).
* Member C: Post-processing (Translating output Semantic Tokens to Jianpu via math mapping).
* Member D: Integration & Deployment (Docker, UI, overall pipeline).

Please provide:
1. **Branching Naming Conventions:** Give specific examples for features, bug fixes, and experiments.
2. **Pull Request (PR) Policy:** Rules for code reviews, resolving conflicts, and merging into `main`.
3. **`.gitignore` Template:** A comprehensive Python/PyTorch/Docker ignore file that prevents uploading massive model weights (`.pth`, `.safetensors`), dataset images, and Jupyter checkpoint caches.

# Output Format
* Output directly usable code blocks for `Dockerfile`, `docker-compose.yml`, `Makefile`, and `.gitignore`.
* Provide concise, bulleted explanations for the GitHub Flow strategy.
* Do not invent code for the model itself; focus purely on the DevOps, environment, and Git infrastructure.