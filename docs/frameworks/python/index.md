---
title: "Python"
summary: "Reference-backed pages about Python runtime mechanisms, execution, and import behavior."
layout: default
confidence: high
sources:
  - logs/index.md
updated: 2026-09-09
---

# Python

- [Python Import System: Cache, Finders, Loaders, and Module Specs](import-system.md) — How Python searches for modules, constructs a `ModuleSpec`, executes a loader, caches the result, and binds names for an import statement.
- [Python Execution Model: Code Blocks, Scopes, and Runtime State](execution-model.md) — How code blocks run in frames, names bind and resolve through namespaces, annotation scopes defer work, and runtime layers share state.
- [Your Guide to the CPython Source Code](cpython-source-code-guide.md) — Historical CPython 3.8 source map from startup and parsing through code objects, frames, objects, the standard library, and tests.
