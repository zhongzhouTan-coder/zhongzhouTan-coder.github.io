---
kind: repository-source
provider: github
clone_url: https://github.com/firecracker-microvm/firecracker.git
repository_url: https://github.com/firecracker-microvm/firecracker
local_checkout: external-repos/firecracker/
commit: f3f65a3425f62bc9cf5f1c81f2963f230ed89f9a
ref: main
inspected: 2026-08-28
checkout_state: clean
---

# firecracker Codebase Source Record

## Reading Scope

- MicroVM sandbox architecture for untrusted AI-agent workloads: API/control plane, KVM/VCPU runtime, jailer process containment, seccomp, cgroups/namespaces, network/storage/vsock boundaries, and lifecycle cleanup

## Important Entry Files

- `docs/design.md` — Architecture, trust zones, device model, networking, storage, and sandboxing rationale
- `docs/jailer.md` — Jailer invocation, chroot, privileges, namespaces, cgroups, and resource limits
- `docs/seccomp.md` — Seccomp threat model and filter generation/installation
- `docs/vsock.md` — Guest-host vsock transport and host integration
- `src/firecracker/src/main.rs` — Firecracker process startup, API server construction, seccomp setup, and lifecycle
- `src/firecracker/src/api_server/mod.rs` — HTTP API server, request parsing, and VMM handoff
- `src/firecracker/src/seccomp.rs` — Default/custom seccomp filter loading and per-thread policy
- `src/vmm/src/rpc_interface.rs` — Preboot/runtime API actions and VMM state transitions
- `src/vmm/src/builder.rs` — MicroVM construction, device attachment, and boot
- `src/vmm/src/vstate/vcpu.rs` — KVM vCPU thread creation, seccomp inheritance, and KVM_RUN loop
- `src/jailer/src/env.rs` — Chroot setup, device preparation, privilege drop, namespace join, and exec
- `src/jailer/src/cgroup.rs` — Cgroup v1/v2 configuration and controller setup
- `src/vmm/src/devices/virtio/vsock/unix/muxer.rs` — Host Unix-socket backend for guest vsock connections
- `src/vmm/src/devices/virtio/net/tap.rs` — TAP-backed guest network boundary
- `src/vmm/src/vmm_config/drive.rs` — Host-backed block device configuration

## Limitations

- Static code reading only; runtime behavior was not executed.
