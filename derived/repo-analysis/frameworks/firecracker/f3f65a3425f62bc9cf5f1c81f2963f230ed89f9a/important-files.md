---
kind: repository-analysis
repository_id: github:firecracker-microvm/firecracker@f3f65a3425f62bc9cf5f1c81f2963f230ed89f9a
commit: f3f65a3425f62bc9cf5f1c81f2963f230ed89f9a
source_record: raw/frameworks/firecracker-codebase--github-f3f65a3425f6.md
generated: 2026-08-28
---

# firecracker Codebase Important Files

## Reader Contract

- **Page type:** code-reading synthesis for readers learning how a microVM becomes an AI-agent sandbox.
- **Audience:** engineers familiar with Linux processes and HTTP, but not necessarily KVM, VirtIO, or seccomp.
- **Question:** Which isolation guarantees does Firecracker provide, how does one control request cross the VMM, and what must an agent platform add around it?
- **Mental model:** Firecracker is a small, one-VM-per-process VMM wrapped by a privileged launcher; an outer agent runner supplies policy and guest services.
- **Lifecycle:** the outer runner prepares images and host resources offline, the jailer constrains the process at launch, the API configures and boots the VM, and runtime device traffic crosses the VMM barriers.
- **Verification boundary:** this is static reading of a clean pinned checkout; no VM, KVM, jailer, network namespace, or guest command was executed.

## Representation Plan

| Reader question | Evidence | Representation | Teaching job |
|---|---|---|---|
| Which layer owns each sandbox guarantee? | `docs/design.md`; `src/jailer/src/env.rs`; `src/firecracker/src/seccomp.rs`; `src/vmm/src/devices/virtio/vsock/unix/muxer.rs` | Original threat-containment figure plus synthesized layer flow | Separate virtualization, process, syscall, I/O, and agent-policy responsibilities. |
| How does a start request travel down and return? | `src/firecracker/src/api_server/*`; `src/vmm/src/rpc_interface.rs`; `src/vmm/src/builder.rs`; `src/vmm/src/vstate/vcpu.rs` | Numbered request round trip | Make parsing, handoff, boot, execution, response, and cleanup concrete. |
| Where does a guest command actually cross the boundary? | `docs/vsock.md`; `src/vmm/src/devices/virtio/vsock/unix/muxer.rs` | Prose plus a boundary trace | Show that vsock is a transport primitive, not an authorization or command-execution policy. |

## Evidence Map

| Docs page | Finding | File | Symbol | Start | End |
|---|---|---|---|---:|---:|
| `docs/frameworks/firecracker/index.md` | process-startup | `src/firecracker/src/main.rs` | `main_exec` | 121 | — |
| `docs/frameworks/firecracker/index.md` | api-parse | `src/firecracker/src/api_server/parsed_request.rs` | `ParsedRequest::try_from` | 67 | 145 |
| `docs/frameworks/firecracker/index.md` | instance-start | `src/firecracker/src/api_server/request/actions.rs` | `parse_put_actions` | 31 | 51 |
| `docs/frameworks/firecracker/index.md` | api-dispatch | `src/firecracker/src/api_server/mod.rs` | `ApiServer::handle_request` | 121 | 145 |
| `docs/frameworks/firecracker/index.md` | api-vmm-bridge | `src/firecracker/src/api_server/mod.rs` | `ApiServer::serve_vmm_action_request` | 147 | 185 |
| `docs/frameworks/firecracker/index.md` | preboot-loop | `src/vmm/src/rpc_interface.rs` | `PrebootApiController::build_microvm_from_requests` | 365 | 429 |
| `docs/frameworks/firecracker/index.md` | start-transition | `src/vmm/src/rpc_interface.rs` | `PrebootApiController::start_microvm` | 627 | 638 |
| `docs/frameworks/firecracker/index.md` | build-boot | `src/vmm/src/builder.rs` | `build_and_boot_microvm` | 373 | 386 |
| `docs/frameworks/firecracker/index.md` | kvm-build | `src/vmm/src/builder.rs` | `build_microvm_for_boot` | 143 | — |
| `docs/frameworks/firecracker/index.md` | vmm-seccomp | `src/vmm/src/seccomp.rs` | `apply_filter` | 93 | 137 |
| `docs/frameworks/firecracker/index.md` | vcpu-thread | `src/vmm/src/vstate/vcpu.rs` | `Vcpu::start_threaded` | 180 | 210 |
| `docs/frameworks/firecracker/index.md` | vcpu-run | `src/vmm/src/vstate/vcpu.rs` | `Vcpu::run` | 217 | 236 |
| `docs/frameworks/firecracker/index.md` | kvm-run | `src/vmm/src/vstate/vcpu.rs` | `Vcpu::run_emulation` | 404 | 431 |
| `docs/frameworks/firecracker/index.md` | jailer-setup | `src/jailer/src/env.rs` | `Env::run` | 646 | — |
| `docs/frameworks/firecracker/index.md` | mount-jail | `src/jailer/src/chroot.rs` | `chroot` | 19 | — |
| `docs/frameworks/firecracker/index.md` | cgroup-setup | `src/jailer/src/cgroup.rs` | `CgroupConfiguration::setup` | 234 | 240 |
| `docs/frameworks/firecracker/index.md` | resource-limits | `src/jailer/src/resource_limits.rs` | `ResourceLimits::install` | 92 | 102 |
| `docs/frameworks/firecracker/index.md` | tap-backend | `src/vmm/src/devices/virtio/net/tap.rs` | `Tap::open_named` | 115 | 151 |
| `docs/frameworks/firecracker/index.md` | block-config | `src/vmm/src/vmm_config/drive.rs` | `BlockDeviceConfig` | 32 | 68 |
| `docs/frameworks/firecracker/index.md` | vsock-config | `src/vmm/src/vmm_config/vsock.rs` | `VsockBuilder::create_unixsock_vsock` | 108 | 115 |
| `docs/frameworks/firecracker/index.md` | vsock-host | `src/vmm/src/devices/virtio/vsock/unix/muxer.rs` | `VsockMuxer::new` | 330 | 352 |
| `docs/frameworks/firecracker/index.md` | vsock-accept | `src/vmm/src/devices/virtio/vsock/unix/muxer.rs` | `VsockMuxer::handle_event` | 360 | 435 |
| `docs/frameworks/firecracker/index.md` | vsock-packet | `src/vmm/src/devices/virtio/vsock/unix/muxer.rs` | `VsockMuxer::send_pkt` | 191 | 253 |
| `docs/frameworks/firecracker/index.md` | response-assembly | `src/firecracker/src/api_server/parsed_request.rs` | `ParsedRequest::convert_to_response` | 186 | 237 |
| `docs/frameworks/firecracker/index.md` | api-shutdown | `src/firecracker/src/api_server_adapter.rs` | `run_with_api` | 250 | 273 |
| `docs/frameworks/firecracker/index.md` | vmm-drop | `src/vmm/src/lib.rs` | `Drop for Vmm` | 765 | 775 |

## Request Round Trip

1. **Boundary entry:** a host-side runner sends configuration requests and an `InstanceStart` action over the Firecracker Unix API socket (`api-parse`, `instance-start`).
2. **API dispatch:** the API thread converts the HTTP request into a typed `VmmAction`, sends it over a channel, signals an `eventfd`, and waits for the VMM response (`api-dispatch`, `api-vmm-bridge`).
3. **Preboot coordination:** the VMM-side preboot loop consumes the configuration sequence, mutates `VmResources`, and returns a response after each action (`preboot-loop`).
4. **Boot transition:** `StartMicroVm` calls the builder; the builder allocates guest memory, creates KVM/vCPU state, loads the kernel and initrd, attaches VirtIO devices, then resumes the vCPUs (`start-transition`, `kvm-build`, `build-boot`).
5. **Guest execution:** each vCPU thread installs its own seccomp program and enters the paused/running state machine around `KVM_RUN`; KVM exits return to Firecracker's device-emulation path (`vcpu-thread`, `vcpu-run`, `kvm-run`).
6. **Response propagation:** the VMM result travels back through the response channel and is converted to an HTTP status/body (`response-assembly`).
7. **Release:** when the VMM signals shutdown, the API adapter joins the API thread and `Vmm`'s destructor shuts down vCPU threads (`api-shutdown`, `vmm-drop`).

The guest-side agent command is intentionally not claimed as a complete repository round trip: this checkout contains the vsock transport, but not an AI-agent command broker inside the guest. The host runner must supply that missing semantic layer.

## Static Reading Notes

- Firecracker's design document treats guest vCPU execution as malicious and describes nested trust zones separated by virtualization, I/O, and jailer barriers.
- The Jailer performs privileged setup before `exec()`ing Firecracker as the requested unprivileged user; files and descriptors made available to the jailed process remain an explicit host responsibility.
- Firecracker's seccomp policy is divided into `vmm`, `api`, and `vcpu` thread categories and is installed per thread before guest-facing work.
- The network backend opens a host TAP device and the design explicitly leaves traffic filtering to the host. I/O rate limiting is not equivalent to an egress allowlist.
- The vsock backend maps a host Unix listener to guest AF_VSOCK traffic and multiplexes connections by port. It does not authenticate the caller or define which guest command a port may run.
