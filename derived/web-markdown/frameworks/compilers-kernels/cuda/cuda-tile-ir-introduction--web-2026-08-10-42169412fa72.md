---
kind: web-extraction
source_url: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html"
final_url: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html"
canonical_url: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/introduction.html"
title: "1. Introduction — Tile IR"
author: ""
published_at: ""
captured_at: "2026-08-10T08:04:34.406Z"
content_sha256: 42169412fa72091fdd4471040ce04d3d8869ef99e499fd181c3a072eae9717e9
renderer: http
extractor: "defuddle@0.13.0 + turndown@7.2.4"
---

1\. Introduction
----------------

This document describes **Tile IR**, a portable, low-level tile virtual machine and instruction set.

Unlike PTX, which models the GPU as a data-parallel single instruction multiple thread (SIMT) processor, **Tile IR** models the GPU as a tile-based processor. In **Tile IR**, each logical thread (tile block) computes over partial fragments (tiles) of multi-dimensional arrays (tensors).

1.1. Scalable Data Parallel Computing with Tiles on GPU
-------------------------------------------------------

The CUDA platform provides portability through CUDA C++ and PTX, allowing code written for older GPU generations to run efficiently on newer ones. These mechanisms enable developers to write software for one family of GPUs and continue to deploy it on future generations.

The rapid evolution of hardware features, such as tensor cores, has increased programming-model complexity, requiring greater expertise and hardware understanding to write portable and performant code. Since Volta, each new GPU generation introduces powerful new hardware features, giving rise to new programming paradigms.

To address these challenges, **Tile IR** introduces a virtual instruction set that enables native programming of the hardware in terms of tiles, allowing developers to write higher-level code that can be efficiently executed across multiple GPUs with minimal changes.

While PTX ensures portability for SIMT programs, **Tile IR** extends the CUDA platform with native support for tile-based programs, allowing developers to focus on partitioning their data-parallel programs into tiles and tile blocks, letting **Tile IR** handle the mapping onto hardware resources such as threads, the memory hierarchy, and tensor cores.

By raising the level of abstraction, **Tile IR** enables users to build new higher-level hardware-specific DSLs, compilers, and frameworks for NVIDIA hardware.

1.2. Goals and Scope
--------------------

**Tile IR** aims to provide a performance-portable programming model and instruction set for tile-based parallel programming.

Core goals:

-   Introduce a data-parallel tile programming abstraction that aligns with programmer intent and acts as a code generation target for DSLs and compilers.

-   Abstract tensor-cores and their programming model to enable hardware innovation without disrupting the NVIDIA ecosystem or customers’ existing software investments.

-   Abstract low-level, architecture-specific details such as CUDA threads and memory hierarchy to ensure programs can be efficiently recompiled for different GPUs.

-   Minimize abstraction overhead. While portability has a cost, **Tile IR** aims to keep this cost low, introducing only a modest performance overhead.

-   Provide user controls and optimization hints to recover peak performance when needed.

-   Provide seamless interoperability with existing CUDA programming models like CUDA C++ and PTX.

**Tile IR** achieves these core goals through the following key components, in order of importance:

1.  A versioned specification of the **Tile IR** abstract machine, including a versioned and portable bytecode representation.

2.  An optimizing compiler for **Tile IR** programs, available as part of the CUDA driver and as a standalone tool in the CUDA toolkit, similar to PTX.

3.  An MLIR dialect that existing compilers can use to target **Tile IR** as a backend compiler target.

1.3. Document Structure
-----------------------

1.  this section.

2.  [Programming Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#section-prog-model) describes the programming model of **Tile IR**.

3.  [Syntax](https://docs.nvidia.com/cuda/tile-ir/latest/sections/syntax.html#section-syntax) defines the syntax of **Tile IR** programs used throughout the specification.

4.  [Binary Format](https://docs.nvidia.com/cuda/tile-ir/latest/sections/bytecode.html#section-bytecode) describes the **Tile IR** byte code and its binary format.

5.  [Type System](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#section-types) defines the type system of **Tile IR**.

6.  [Semantics](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#section-semantics) describes the semi-formal semantics of **Tile IR**.

7.  [Memory Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#section-memory-model) describes the **Tile IR** memory model.

8.  [Operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#section-operations) a listing of the full set of **Tile IR** operations.

9.  [Debug Info](https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html#section-debug-info) describes the debug information format for **Tile IR** programs.

10.  [Stability](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#section-stability) describes the stability guarantees of the **Tile IR**.

11.  [Appendix](https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#section-appendix) contains relevant reference materials, including full programs and references.
