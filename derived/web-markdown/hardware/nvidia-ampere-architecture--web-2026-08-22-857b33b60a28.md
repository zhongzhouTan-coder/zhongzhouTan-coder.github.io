---
kind: web-extraction
source_url: "https://www.nvidia.com/en-us/data-center/ampere-architecture/"
final_url: "https://www.nvidia.com/en-us/data-center/ampere-architecture/"
canonical_url: "https://www.nvidia.com/en-us/data-center/ampere-architecture/"
title: "NVIDIA Ampere Architecture: The Heart of the Modern Data Center"
author: "NVIDIA"
published_at: ""
captured_at: "2026-08-22T10:04:12.019Z"
content_sha256: 857b33b60a286ee113c160d66e2470ddb307e8489f49170d6da54849499e5e9b
renderer: http
extractor: "defuddle@0.13.0 + turndown@7.2.4"
---

NVIDIA Ampere Architecture
--------------------------

The heart of the world’s highest-performing, elastic data centers.

The Core of AI and HPC in the Modern Data Center
------------------------------------------------

Solving the world’s most important scientific, industrial, and business challenges with AI and [HPC](https://www.nvidia.com/en-us/high-performance-computing/). Visualizing complex content to create cutting-edge products, tell immersive stories, and reimagine cities of the future. Extracting new insights from massive datasets. The NVIDIA Ampere architecture, designed for the age of elastic computing, rises to all these challenges, providing unmatched acceleration at every scale.

Groundbreaking Innovations
--------------------------

Crafted with 54 billion transistors, the [NVIDIA Ampere architecture](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) is the largest 7 nanometer (nm) chip ever built and features six key groundbreaking innovations.

### Third-Generation Tensor Cores

First introduced in the NVIDIA Volta™ architecture, NVIDIA Tensor Core technology has brought dramatic speedups to AI, bringing down training times from weeks to hours and providing massive acceleration to inference. The NVIDIA Ampere architecture builds upon these innovations by bringing new precisions—Tensor Float 32 (TF32) and floating point 64 (FP64)—to accelerate and simplify AI adoption and extend the power of Tensor Cores to HPC.

TF32 works just like FP32 while delivering speedups of up to 20X for AI without requiring any code change. Using [NVIDIA Automatic Mixed Precision](https://developer.nvidia.com/automatic-mixed-precision), researchers can gain an additional 2X performance with automatic mixed precision and FP16 by adding just a couple of lines of code. And with support for bfloat16, INT8, and INT4, Tensor Cores in NVIDIA Ampere architecture Tensor Core GPUs create an incredibly versatile accelerator for both AI training and inference. Bringing the power of Tensor Cores to HPC, [A100](https://www.nvidia.com/en-us/data-center/a100/) and [A30 GPUs](https://www.nvidia.com/en-us/data-center/products/a30-gpu/) also enable matrix operations in full, IEEE-certified, FP64 precision.

### Multi-Instance GPU (MIG)

Every AI and HPC application can benefit from acceleration, but not every application needs the performance of a full GPU. Multi-Instance GPU (MIG) is a feature supported on [A100](https://www.nvidia.com/en-us/data-center/a100/) and [A30 GPUs](https://www.nvidia.com/en-us/data-center/products/a30-gpu/) that allows workloads to share the GPU. With MIG, each GPU can be partitioned into multiple GPU instances, fully isolated and secured at the hardware level with their own high-bandwidth memory, cache, and compute cores. Now, developers can access breakthrough acceleration for all their applications, big and small, and get guaranteed quality of service. And IT administrators can offer right-sized GPU acceleration for optimal utilization and expand access to every user and application across both bare-metal and virtualized environments.

### Third-Generation NVLink

Scaling applications across multiple GPUs requires extremely fast movement of data. The third generation of NVIDIA® NVLink® in the NVIDIA Ampere architecture doubles the GPU-to-GPU direct bandwidth to 600 gigabytes per second (GB/s), almost 10X higher than PCIe Gen4. When paired with the latest generation of [NVIDIA NVSwitch™](https://www.nvidia.com/en-us/data-center/nvlink/), all GPUs in the server can talk to each other at full NVLink speed for incredibly fast data transfers.

[NVIDIA DGX™A100](https://www.nvidia.com/en-us/data-center/dgx-a100/) and servers from other leading computer makers take advantage of NVLink and NVSwitch technology via [NVIDIA HGX™ A100](https://www.nvidia.com/en-us/data-center/hgx/) baseboards to deliver greater scalability for HPC and AI workloads.

### Structural Sparsity

Modern AI networks are big and getting bigger, with millions and in some cases billions of parameters. Not all of these parameters are needed for accurate predictions and inference, and some can be converted to zeros to make the models “sparse” without compromising accuracy. [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/) can provide up to 2X higher performance for sparse models. While the sparsity feature more readily benefits AI inference, it can also be used to improve the performance of model training.

### Second-Generation RT Cores

The NVIDIA Ampere architecture’s second-generation RT Cores in the [NVIDIA A40](https://www.nvidia.com/en-us/data-center/a40/) deliver massive speedups for workloads like photorealistic rendering of movie content, architectural design evaluations, and virtual prototyping of product designs. RT Cores also speed up the rendering of ray-traced motion blur for faster results with greater visual accuracy and can simultaneously run ray tracing with either shading or denoising capabilities.

### Smarter and Faster Memory

[A100](https://www.nvidia.com/en-us/data-center/a100/) brings massive amounts of compute to data centers. To keep those compute engines fully utilized, it has a class-leading 2 terabytes per second (TB/sec) of memory bandwidth, more than double the previous generation. In addition, A100 has significantly more on-chip memory, including a 40 megabyte (MB) level 2 cache—7X larger than the previous generation—to maximize compute performance.

Optimized For Scale
-------------------

NVIDIA GPU and NVIDIA converged accelerator offerings are purpose built to deploy at scale, bringing networking, security, and small footprints to the cloud, data center, and edge.

### Power Optimized for Any Server

Offering the smallest footprint in the portfolio, the [NVIDIA A2 GPU](https://www.nvidia.com/en-us/data-center/products/a2/) is optimized for inference workloads and deployments in entry-level servers constrained by space and thermal requirements, such as 5G edge and industrial environments. A2 delivers a low-profile form factor operating in a low-power envelope, from a thermal design power (TDP) of 60W down to 40W, making it ideal for any server.

### Unified Compute and Network Acceleration

In [NVIDIA converged accelerators](https://www.nvidia.com/en-us/data-center/products/converged-accelerator/), the NVIDIA Ampere architecture and the [NVIDIA BlueField®-2](https://www.nvidia.com/en-us/networking/products/data-processing-unit/) data processing unit (DPU) come together to bring unprecedented performance with enhanced security and networking for GPU-powered workloads in edge computing, telecommunications, and network security. BlueField-2 combines the power of the NVIDIA ConnectX®-6 Dx with programmable Arm® cores and hardware offloads for software-defined storage, networking, security, and management. NVIDIA converged accelerators enable a new level of data center efficiency and security for network-intensive, GPU-accelerated workloads.

### Density Optimized Design

NVIDIA A16 GPU comes in a quad-GPU board design that’s optimized for user density and, combined with [NVIDIA Virtual PC (vPC) software](https://www.nvidia.com/en-us/data-center/virtual-pc-apps/), enables graphics-rich virtual PCs accessible from anywhere. Deliver increased frame rate and lower end user latency versus CPU-only VDI with NVIDIA A16, resulting in more responsive applications and a user experience that’s indistinguishable from a native PC.

### Secure Deployments

Secure deployments are critical for enterprise business operations. The NVIDIA Ampere architecture optionally delivers secure boot through trusted code authentication and hardened rollback protections to protect against malicious malware attacks, preventing operational losses and ensuring workload acceleration.

Inside the NVIDIA Ampere Architecture

Explore the cutting-edge technologies of the NVIDIA Ampere architecture.
