---
title: "Monkey-Patching"
summary: "Runtime replacement, deletion, or redirection of functions, attributes, modules, or environment lookups so code executes against controlled behavior without editing the original source."
tooltip: "Monkey-patching changes what code sees at runtime by replacing functions, attributes, imports, or environment values. It is useful in tests and plugin systems, but it is fragile because it depends on where code performs the lookup."
layout: default
confidence: high
category: general
sources:
  - raw/frameworks/pytest-monkeypatch-documentation--web-2026-08-03-71ec0c060fa3.html
  - raw/frameworks/pytest-monkeypatch-documentation--web-2026-08-03-71ec0c060fa3.metadata.json
  - derived/web-markdown/frameworks/pytest-monkeypatch-documentation--web-2026-08-03-71ec0c060fa3.md
aliases:
  - monkey patch
  - monkeypatch
  - monkey-patch
  - monkeypatching
appears_in:
  - docs/benchmarks/agent-eval/deepswe-v1-1/index.md
  - docs/frameworks/triton-ascend/index.md
  - docs/frameworks/vllm-ascend/architecture.md
  - docs/frameworks/vllm-ascend/index.md
updated: 2026-08-03
---

# Monkey-Patching

**Monkey-Patching** is runtime replacement, deletion, or redirection of functions, attributes, modules, or environment lookups so code executes against controlled behavior without editing the original source.

## Why It Exists

Some code depends on global state, external services, filesystem layout, import resolution, or library internals that are inconvenient or unsafe to exercise directly. Monkey-patching lets a test or plugin intercept those lookups and substitute a controlled behavior close to the call site.

## How It Works

Monkey-patching changes the object that the target code will look up at runtime. That can mean replacing a method on a class, swapping a module function, deleting a network request entry point, overriding an environment variable, or adjusting `sys.path` before an import happens.

In pytest, the `monkeypatch` fixture packages these operations behind helpers such as `setattr`, `delattr`, `setitem`, `setenv`, and `syspath_prepend`, then automatically undoes the changes when the test or fixture finishes. The important detail is lookup location: patch the symbol where the code under test resolves it, not merely where it was originally defined.

## Tradeoffs

Monkey-patching is powerful but brittle. It depends on internal names and lookup paths staying stable, so refactors can silently break the patch. Broad patches can also hide integration problems or leak state across tests if cleanup is incomplete.

## Common Confusions

- **Monkey-patching vs. dependency injection:** Dependency injection changes dependencies through explicit parameters or constructors; monkey-patching changes them implicitly at runtime.
- **Monkey-patching vs. mocking:** Mocking usually emphasizes test doubles and assertions about calls; monkey-patching is the mechanism that replaces the real object or behavior.
- **pytest `monkeypatch` vs. general monkey-patching:** The pytest fixture is a disciplined helper that applies and reverts monkey-patches automatically; the broader concept is language- and framework-independent.

## Where It Appears

- [Monkeypatching/mocking modules and environments — pytest documentation](https://docs.pytest.org/en/6.2.x/monkeypatch.html) — Documents a structured testing fixture for patching attributes, dictionary items, environment variables, and import paths with automatic cleanup.
- [DeepSWE v1.1: Execution and Scoring Changes](../benchmarks/agent-eval/deepswe-v1-1/index.md) — Uses monkey-patching as the example of a benchmark shortcut that committed-patch-only grading is meant to block.
- [Triton Ascend: Ascend NPU Backend for Triton](../frameworks/triton-ascend/index.md) — Describes how triton-ascend injects backend-specific behavior into upstream Triton without maintaining a fork.
- [vLLM Ascend](../frameworks/vllm-ascend/index.md) — The hub page summarizes monkey-patches as one of the key integration mechanisms in the Ascend port.
- [vLLM-Ascend Architecture: How the Ascend NPU Port Integrates with vLLM](../frameworks/vllm-ascend/architecture.md) — Explains monkey-patches as the layer that adapts CUDA-coupled vLLM internals to Ascend.
