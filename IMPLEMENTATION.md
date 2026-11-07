# Implementation ownership contract

Distribution: `lilax`; import: `lilax`; source root: `src/lilax`.

Original computational modules migrate from `src/maxtext/<path>` to `src/lilax/_backend/<path>`.
All internal `maxtext` imports become `lilax._backend` imports. Original source remains unchanged.
Migrated code must not depend on the original `maxtext` or `dependencies` packages at runtime.
Original numerical behavior, defaults, restrictions, parameter/state layouts, and external type identities remain intact.
No module aliases, synthetic argv, subprocess legacy CLI, dependency shims, new lazy imports, or fake implementations.
Preserve existing source-local accelerator imports where moving them globally would change platform dependency behavior; this does not permit new lazy imports, missing-dependency fallbacks, shims, or stubs.

Public modules use these canonical names:

- `configuration`: `MaxTextConfig`, `AutoConfig`, typed backend options; no device/environment initialization.
- `runtime`: `Runtime`, explicit lifecycle and platform/distributed initialization.
- `modeling.models`: `AutoModelForCausalLM`, native model objects and real weight import/export through `modeling.hub`.
- `modeling.hub`: supported-model capability registry and consistent Hub resolution/conversion.
- `data_pipeline.api`: native data backends and direct Hugging Face Dataset/IterableDataset adaptation.
- `trainers`: `Trainer`, `TrainingArguments`, SFT/DPO/GRPO/distillation classes and configs.
- `peft`: native adapter APIs preserving supported combinations.
- `generation`, `serving`, `evaluation`, `benchmarks`, `diagnostics`, `launch` and `provisioning`: specialized callable Python capabilities. `lilax launch` owns execution-profile launch.

Peer owners (disjoint ownership, not permanent agent quotas):

- Foundation Terra-high coordinator: configuration/runtime/packaging workers, `_backend/configs/**`, package-required root assets/resources, package `__init__.py` integration.
- Models Terra-high coordinator: `_backend/models`, `_backend/layers`, `_backend/kernels`, `_backend/multimodal`, public `modeling`.
- Training Terra-high coordinator: `_backend/trainers`, `_backend/training_engine`, `_backend/optimizers`, `_backend/integration/tunix`, public `trainers`, `peft`.
- Infrastructure Terra-high coordinator: `_backend/common`, `_backend/utils`, `_backend/input_pipeline`, `_backend/checkpoint_conversion`, public `data_pipeline`, checkpoint operations.
- Inference Terra-high coordinator: `_backend/inference`, `_backend/integration/vllm`, `_backend/eval`, `_backend/experimental`, `_backend/m3`, `_backend/diffusion`, public `generation`, `serving`, `evaluation`, benchmarking and launch capabilities except checkpoint operations.

Each owner may split its independent files among workers and must publish exact subownership before editing.
The main agent coordinates these peer owners. Each owner retains its own Terra/Luna children; foundation does not manage other owners' workers.
The shared pool is at most twenty Terra including the main agent and all coordinators, and twenty Luna. Sol is not used. Released capacity is rebalanced through main; initial assignments are not permanent quotas.
Cross-owner edits require coordination. Do not write another owner's files merely because they are not present yet.
Configuration exposes original option names through typed fields/attributes compatible with native consumers.
Runtime owns configuration derivation requiring hardware; configuration construction itself stays inert.
Model/trainer/runtime handoffs must be agreed from actual native call signatures, not guessed.

Source-grounded integration contracts:

- `Runtime` owns resolved native configuration and training mesh; public model objects own `native_model` and `runtime`.
- Native model creation is `model_creation_utils.from_config(config, devices=None, mesh=None, *, model_mode="train", rngs=None, quant_mode_str="train")`.
- Native training engine accepts an optional keyword-only `model=None`; a supplied native model must be retained, not recreated.
- Distributed initialization preserves `max_utils.maybe_initialize_jax_distributed_system(raw_keys)`.
- Training mesh creation preserves `maxtext_utils.get_mesh_from_config(config, devices)` and elastic retry semantics; MaxEngine retains its distinct mesh creation.
- Evaluation retains `logical_axis_rules_for_eval`, distinct from training `logical_axis_rules`.
- Public configuration remains inert. Native dtype conversion and hardware/environment-dependent derivation run only through explicit runtime initialization.
- TPU post-training vLLM plugin registration is a separate distribution under `_backend/assets/integration/vllm`; installing the base or CUDA profile does not register its entry point.
- `resolve_runtime_config(public_config)` in `_backend.configs.runtime` returns native `HyperParameters`; native configuration models and the derived validator are authored, including the one-shot HLO context handoff. Runtime behavior remains unverified.
- `Runtime.public_config` retains the inert input identity; `Runtime.config` becomes the resolved native configuration. Default public model construction initializes runtime internally.

Every rewriter reads the original implementation for every migrated feature, applicable AGENTS.md, and exact dependency requirement profiles.
Prior inventory reports guide discovery but never replace source reading.
Use official `apply_patch` for every edit. No custom copying/patching scripts.
No tests, test files, checks, builds, lint, compilation, validation scripts, git diff/status, commits, branches, PRs, or pushes are authorized.
No global interpreter/environment use or installs. Necessary installations are permitted only in a project virtualenv. Temporary artifacts belong only in `~/tmp/lilax`.
The user authorized `/home/shadowlilac/.agents/skills/poteto-mode/SKILL.md` for all workers. Read it; explicit model, throughput, no-tests, no-commits, and source-reading requirements override skill defaults.
Existing source comments may be preserved; do not author new inline comments or copyright headers.
Keep code compact, readable, source-faithful, and separated by responsibility.

Coverage entries map source evidence and feature to destination owner, real callable public API, implementation/integration state, and `verification: not run`.
Unimplemented items remain explicitly unimplemented; file presence is not parity or execution verification.
