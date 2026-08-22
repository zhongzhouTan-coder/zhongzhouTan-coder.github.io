---
kind: web-extraction
source_url: "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/"
final_url: "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/"
canonical_url: "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/"
title: "NVIDIA Blackwell Architecture"
author: "NVIDIA"
published_at: ""
captured_at: "2026-08-22T10:03:46.135Z"
content_sha256: b064f05f1533fecfe618292d38bdab63db1c79bbe7e2281175ddef49a34f463f
renderer: http
extractor: "defuddle@0.13.0 + turndown@7.2.4"
---

The engine behind AI factories for the age of AI reasoning—now in full production.

Breaking Barriers in Accelerated Computing and Generative AI
------------------------------------------------------------

Explore the groundbreaking advancements the NVIDIA Blackwell architecture brings to generative AI and accelerated computing. Building upon generations of [NVIDIA technologies](https://www.nvidia.com/en-us/technologies/), NVIDIA Blackwell defines the next chapter in generative AI with unparalleled performance, efficiency, and scale.

Look Inside the Technological Breakthroughs
-------------------------------------------

### A New Class of AI Superchip

NVIDIA Blackwell-architecture GPUs pack 208 billion transistors and are manufactured using a custom-built TSMC 4NP process. All NVIDIA Blackwell products feature two reticle-limited dies connected by a 10 terabytes per second (TB/s) chip-to-chip interconnect in a unified single GPU.

#### Second-Generation Transformer Engine

The second-generation Transformer Engine uses custom [NVIDIA Blackwell Tensor Core](https://www.nvidia.com/en-us/data-center/tensor-cores/) technology combined with NVIDIA TensorRT™-LLM and NeMo™ Framework innovations to accelerate inference and training for large language models (LLMs) and [Mixture-of-Experts (MoE) models](https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/). NVIDIA Blackwell Tensor Cores add new precisions, including new community-defined microscaling formats, giving high accuracy and ease of replacement for larger precisions.

NVIDIA Blackwell Ultra Tensor Cores are supercharged with 2X the attention-layer acceleration and 1.5X more AI compute FLOPS compared to NVIDIA Blackwell GPUs. The NVIDIA Blackwell Transformer Engine utilizes fine-grain scaling techniques called micro-tensor scaling, to optimize performance and accuracy enabling 4-bit floating point (FP4) AI. This doubles the performance and size of next-generation models that memory can support while maintaining high accuracy.

### Secure AI

NVIDIA Blackwell includes NVIDIA Confidential Computing, which protects sensitive data and AI models from unauthorized access with strong hardware-based security. NVIDIA Blackwell is the first TEE-I/O capable GPU in the industry, while providing the most performant confidential compute solution with TEE-I/O capable hosts and inline protection over NVIDIA NVLink™. NVIDIA Blackwell Confidential Computing delivers nearly identical throughput performance compared to unencrypted modes. Enterprises can now secure even the largest models in a performant way, in addition to protecting AI intellectual property (IP) and securely enabling confidential AI training, inference, and federated learning.

#### NVLink and NVLink Switch

Unlocking the full potential of exascale computing and trillion-parameter AI models hinges on the need for swift, seamless communication among every GPU within a server cluster. The fifth-generation of NVIDIA NVLink interconnect can scale up to 576 GPUs to unleash accelerated performance for trillion- and multi-trillion parameter AI models.

The NVIDIA NVLink Switch Chip enables 130TB/s of GPU bandwidth in one 72-GPU NVLink domain (NVL72) and delivers 4X bandwidth efficiency with NVIDIA Scalable Hierarchical Aggregation and Reduction Protocol (SHARP)™ FP8 support. The NVIDIA NVLink Switch Chip supports clusters beyond a single server at the same impressive 1.8TB/s interconnect. Multi-server clusters with NVLink scale GPU communications in balance with the increased computing, so NVL72 can support 9X the GPU throughput than a single eight-GPU system.

### Decompression Engine

Data analytics and database workflows have traditionally relied on CPUs for compute. Accelerated data science can dramatically boost the performance of end-to-end analytics, speeding up value generation while reducing cost. Databases, including Apache Spark, play critical roles in handling, processing, and analyzing large volumes of data for data analytics.

NVIDIA Blackwell’s Decompression Engine and ability to access massive amounts of memory in the [NVIDIA Grace™ CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) over a high-speed link—900 gigabytes per second (GB/s) of bidirectional bandwidth—accelerate the full pipeline of database queries for the highest performance in data analytics and data science with support for the latest compression formats such as LZ4, Snappy, and Deflate.

#### Reliability, Availability, and Serviceability (RAS) Engine

NVIDIA Blackwell adds intelligent resiliency with a dedicated Reliability, Availability, and Serviceability (RAS) Engine to identify potential faults that may occur early on to minimize downtime. NVIDIA’s AI-powered predictive-management capabilities continuously monitor thousands of data points across hardware and software for overall health to predict and intercept sources of downtime and inefficiency. This builds intelligent resilience that saves time, energy, and computing costs.

NVIDIA’s RAS Engine provides in-depth diagnostic information that can identify areas of concern and plan for maintenance. The RAS engine reduces turnaround time by quickly localizing the source of issues and minimizes downtime by facilitating effective remediation.

### NVIDIA Blackwell Ultra Delivers up to 50x Better Performance and 35x Lower Cost for Agentic AI

Built to accelerate the next generation of agentic AI, NVIDIA Blackwell Ultra delivers breakthrough inference performance with dramatically lower cost. Cloud providers such as Microsoft, CoreWeave, and Oracle Cloud Infrastructure are deploying NVIDIA GB300 NVL72 systems at scale for low-latency and long-context use cases, such as agentic coding and coding assistants.

This is enabled by deep co-design across NVIDIA Blackwell, NVLink™, and NVLink Switch for scale-out; NVFP4 for low-precision accuracy; and NVIDIA Dynamo and TensorRT™ LLM for speed and flexibility—as well as development with community frameworks SGLang, vLLM, and more.

NVIDIA Blackwell Products
-------------------------

### NVIDIA GB300 NVL72

The NVIDIA GB300 NVL72 delivers unparalleled AI reasoning inference performance, featuring 65X more AI compute than Hopper systems.

### NVIDIA DGX SuperPOD

NVIDIA DGX SuperPOD™ is a turnkey AI data center solution that delivers leadership-class accelerated infrastructure with scalable performance for the most demanding AI training and inference workloads.

### NVIDIA RTX PRO in the Data Center

Deliver powerful AI and graphics acceleration, essential enterprise features, and the flexibility to handle a wide range of workloads, from agentic and physical AI to visual computing and virtual workstations accelerated by NVIDIA RTX PRO™ data center GPUs.

### NVIDIA RTX PRO Workstations

Bring the latest breakthroughs in AI, ray tracing, and neural graphics technology to power the most innovative workflows in design, engineering, and beyond with NVIDIA RTX PRO GPUs.

### NVIDIA DGX Station

Unlike any AI desktop computer before, this system features NVIDIA Blackwell GPUs, the Grace CPU Superchip, and large coherent memory, delivering unparalleled compute performance.

### NVIDIA DGX Spark

A compact, personal AI supercomputer with the NVIDIA GB10 Grace Blackwell Superchip, delivering high-performance AI capabilities and support for models up to 200 billion parameters.

### NVIDIA HGX B300

NVIDIA HGX™ B300 is built for the age of AI reasoning with enhanced compute and increased memory.

### NVIDIA GB200 NVL72

The NVIDIA GB200 NVL72 connects 36 NVIDIA Grace CPUs and 72 NVIDIA Blackwell GPUs in a rack-scale, liquid-cooled design.

### NVIDIA GB200 NVL4

Purpose-built for scientific computing, the NVIDIA GB200 NVL4 unlocks the future of converged high-performance computing and AI.

![Image](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/dgx-platform/nvidia-dgx-spark-bm-uf-bottom-p.jpg,%20/content/dam/en-zz/Solutions/data-center/dgx-platform/nvidia-dgx-spark-bm-uf-bottom-p@2x.jpg)

NVIDIA DGX Spark
----------------

DGX Spark brings the power of NVIDIA Grace Blackwell™ to developer desktops. The GB10 Superchip, combined with 128 GB of unified system memory, lets AI researchers, data scientists, and students work with AI models locally with up to 200 billion parameters.

Unlock Real-Time, Trillion-Parameter Models With the NVIDIA GB200 NVL72
-----------------------------------------------------------------------

The NVIDIA GB200 NVL72 connects 36 GB200 Grace Blackwell Superchips with 36 Grace CPUs and 72 Blackwell GPUs in a rack-scale design. The GB200 NVL72 is a liquid-cooled solution with a 72-GPU NVLink domain that acts as a single massive GPU—delivering 30X faster real-time inference for trillion-parameter large language models.

NVIDIA NVFP4 Technical Blog
---------------------------

Learn how NVIDIA’s new 4‑bit NVFP4 quantization for pretraining unlocks huge improvements in training LLMs at scale and overall infrastructure efficiency.
