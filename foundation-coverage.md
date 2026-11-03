# Foundation coverage

Statuses describe source integration, not verified behavior. No verification was run.

| Original source | Destination | Real callable or data | Status |
| --- | --- | --- | --- |
| `src/maxtext/configs/types.py`, `src/maxtext/configs/pyconfig.py` | `lilax.configuration.schema` | `MaxTextConfig`, `get_source_values`, `get_source_inputs` | Inert configuration includes custom mesh rules, pipeline defaults, validations and layout changes, FP8 pipeline history sharding, parallelism lists, and source-input provenance. Device-dependent batch-size validation remains native. These paths are authored; exhaustive field/validator parity and execution behavior are not established. |
| `src/maxtext/configs/pyconfig.py`, `src/maxtext/configs/base.yml` | `lilax.configuration.loading` | `AutoConfig.from_pretrained`, `load_config_values` | Implemented; runtime behavior remains unverified. |
| `src/maxtext/utils/globals.py`, `src/maxtext/checkpoint_conversion/` | `lilax.modeling.hub` | `ModelCapabilities`, `get_model_capabilities`, `NATIVE_MODELS`, `HF_IMPORT_MODELS`, `HF_EXPORT_MODELS`, `HF_REPOS`, `MODEL_CONFIG_NAMES` | Integrated into the package root; capability data is maintained in the migrated backend. |
| `src/maxtext/utils/globals.py` and Hugging Face Hub | `lilax.modeling.hub` | `resolve_model_name`, `resolve_model_snapshot` | Implemented in the explicit public submodule. |
| `src/maxtext/configs/pyconfig.py` | `lilax.runtime`, `lilax._backend.configs.runtime` | `Runtime.initialize`, `resolve_runtime_config` | Authored with native models, one-shot HLO context, and resolved RL nested-value handoff; runtime behavior remains unverified. |
| `src/maxtext/utils/model_creation_utils.py:744`, `src/maxtext/trainers/post_train/rl/train_rl.py` | `lilax.runtime` | `Runtime.for_rl(config, *, devices=None)` | Authored pre-model partition lifecycle using Infra's native helper; retains original inert config identity, independently resolves trainer and sampler topology, and builds the trainer mesh on its partition. Eager dependency imports and runtime behavior remain unverified. |
| `src/maxtext/utils/model_creation_utils.py` | `lilax.modeling.models` | `AutoModelForCausalLM.from_config`, `AutoModelForCausalLM.from_pretrained` | Factory, decoder dispatch for the reviewed families, and multimodal encoder callers were source-traced without a demonstrated missing handoff. Arbitrary transformed graph reconstruction is not established; integration and execution remain unverified. |
| `src/maxtext/checkpoint_conversion/to_maxtext.py`, `src/maxtext/checkpoint_conversion/to_huggingface.py` | `lilax.modeling.models` | `AutoModelForCausalLM.from_pretrained`, `MaxTextModel.save_pretrained` | Automatic CPU conversion retains the source-narrow projection and disables multimodal; explicit conversion preserves the public shape, architecture, and multimodal configuration with CPU-specific overrides. Lazy multimodal conversion remains rejected. Native save uses captured pre-pipeline layout and the default eval selector without mutating the live config; explicit-key metadata is retained and the credential is excluded. Conversion, parent restore, and lifecycle integration remain unverified. |
| `src/maxtext/input_pipeline/input_pipeline_interface.py` | `lilax.data_pipeline.api` | `create_data_iterator` | Exposed in the explicit public submodule; runtime behavior remains unverified. |
| `src/maxtext/trainers/pre_train/train.py` | `lilax.trainers` | `Trainer.train`, `Trainer.evaluate`, `Trainer.save_model`, `train_pretraining` | Train, evaluation, save, and selected-step resume surfaces are authored; supplied-model resize requires a compatible factory. Reviewed lifecycle bodies have no confirmed omissions; full caller/body coverage and runtime integration are not established. |
| `src/maxtext/trainers/post_train/sft/train_sft.py`, `src/maxtext/trainers/post_train/dpo/train_dpo.py` | `lilax.trainers.post_training.sft`, `lilax.trainers.post_training.dpo` | `SFTTrainer.train`, `DPOTrainer.train` | Native and Tunix lifecycles are authored; reviewed bodies have no confirmed omissions. Cooperative Lilax-owned Tunix XProf capture stops at the next data fetch or training exit; only windows completed at the next fetch are handed off for artifact upload. Full caller/body coverage, capture/upload behavior and runtime integration are not established. |
| `src/maxtext/trainers/post_train/rl/train_rl.py` | `lilax.trainers.reinforcement` | `GRPOTrainer.from_config`, `GRPOTrainer.train` | Config-first factory uses `Runtime.for_rl` before model creation and preserves native reference/actor load-versus-clone selection; explicit supplied-reference/actor use remains separate. Reviewed lifecycle bodies have no confirmed omissions; full caller/body coverage and runtime integration are not established. |
| `src/maxtext/trainers/post_train/distillation/train_distill.py` | `lilax.trainers.post_training.distillation` | `DistillationTrainer.train` | Student/teacher construction reloads YAML with overrides and the job entry supports resume; teacher capture remains architecture-dependent. Reviewed lifecycle bodies have no confirmed omissions; full caller/body coverage and runtime integration are not established. |
| `src/maxtext/configs/types.py` | `lilax._backend.configs.types` | `MaxTextConfig`, `RLConfig` | Models and derived-value validators authored through the source method's final return; source parity and runtime behavior remain unverified. |

The helper span `src/maxtext/configs/types.py:4123–4306` was source-compared with
`lilax/src/lilax/_backend/configs/types.py:4124–4297`; executable branches, defaults, order,
and messages match. This does not establish runtime behavior or top-level validator parity.

Public schema class metadata restoration against `src/maxtext/configs/types.py` was reported
complete by the sequential writers for `RunInfo`, `Checkpointing`, `OrbaxStorage`, `EmergencyCheckpointing`,
`DataTypes`, `MoBa`, `MlaAttention`, `CompressedAttention`, `AttentionIndexer`,
`Llama4Attention`, `Quantization`, `ModelArchitecture`, `Attention`, `MTP`, `LogitsAndLoss`,
`DeepSeekMoE`, `Ming`, `Qwen3Next`, `Qwen3_8FlashNext`, `Engram`, `DcnParallelism`,
`IciParallelism`, `PipelineParallelism`, `RematAndOffload`, `SplashAttention`, `MoEGeneral`,
`MoEKernels`, `HardwareAndMesh`, `LayoutAndSharding`, `Tokenizer`, `DatasetGeneral`,
`TfdsDataset`, `HfDataset`, `GrainDataset`, `MMapDataset`, `OlmoGrainDataset`, `DPO`,
`FineTuning`, `LoRA`, `Distillation`, `TrainingLoop`, `ManifoldConstrainedHyperConnections`,
`DilocoParams`, `Optimizer`, `AdamW`, `Muon`, `PositionalEmbedding`, `Rope`, `YarnRope`,
`InferenceGeneral`, `Decoding`, `InferenceLayout`, `InferenceServer`, `InferenceBenchmark`,
`PrefixCaching`, `AOT`, `DevelopmentAndDebugging`, `ProfilingCore`, `Profiling`, `HloDump`,
`Metrics`, `ManagedMLDiagnostics`, `Goodput`, `ElasticTraining`, `GcpMonitoring`, `Tensorboard`,
`MultimodalGeneral`, `VisionTower`,
`VisionProjector`, `AudioEncoder`, `Multimodal`, `RLCluster`, `VLLM`, `RL`, `RLDataset`,
`RLEvaluation`, `RLReward`, `RLSpecialTokens`, `DerivedValues`, `MaxTextConfig`, and `RLConfig`.
Independent source-only reviews subsequently reported matching original class docstrings and
corresponding `Field` descriptions for these classes. This is not a runtime parity claim and
does not establish equivalence of annotations, defaults, or validators.

The root does not import hardware-dependent model, runtime, or trainer modules. Explicit submodule imports are not a claim that those APIs are currently usable.

Common package metadata now declares the root's direct Jinja2, Pydantic, PyYAML, OmegaConf,
and Hugging Face Hub dependencies. Hub's constraint and the unpinned Pydantic/OmegaConf
declarations follow the original base requirements; Jinja2/PyYAML constraints follow the
original generated profiles. No accelerator dependencies were added to common metadata.
Root import execution and dependency resolution have not been run.

Common metadata also declares unpinned `librosa` and `soundfile` for the authored direct
multimodal audio imports. Ming's source resampling does not import `torchaudio`; none was added.

The core `trainers` namespace no longer directly imports optional post-training, RL, experimental
GRPO, or engine AOT modules. Core input pipelines still eagerly import TensorFlow and TFDS;
decoupled profiling also needs the real cloud diagnostics dependencies. Existing `tfds` and
`diagnostics` extras supply those declared dependencies; import closure remains unverified.
Experimental GRPO and engine compilation use `trainers.experimental_grpo` and `trainers.engine_aot`.
Ordinary public model imports now eagerly reach native MaxEngine and JetStream through
`generation.engine`. The `generation` extra declares only the exact original JetStream archive;
the inert root remains separate. Dependency resolution and model import behavior remain unverified.
Optional source-pinned post-training dependencies still conflict with generated JAX, Qwix, and
Tokamax requirements; the source's separate `--no-deps` installation does not establish compatibility.
See the README's dependency contract.

`MaxTextConfig.get_source_values()` returns a fresh full mapping with the original learning-rate
schedule selector, pre-pipeline layout, and default evaluation selector restored. Derived `get_keys()`
is unchanged. Envelope serialization retains the selector in its existing values field; legacy
derived-positive dumps cannot recover lost auto provenance. `resolve_initialized_config` prepares
and validates effective overrides without runtime initialization, but native validation may query
hardware. Models has adopted the source mapping for temporary save serialization with actual
construction `scan_layers`. TrainingArguments now consumes the initialized resolver for overrides
without mutating runtime/public config identities; specialized RL configuration remains separate.
No execution verification was run.

`MaxTextConfig.get_source_inputs()` returns a deep-copied `(requested_root, raw_constructor_overrides)`
pair captured before YAML loading. A root of `None` within the pair selects the default YAML;
an unavailable pair is returned as `None`. Ordinary copies retain these inputs; updates to the
root selector or derived fields invalidate them. Hub/snapshot and deserialized flat configurations
do not recover these layers. The existing envelope and model-save formats are unchanged.
`DistillationTrainer.from_config` now consumes the accessor, with explicit `config_file` taking
precedence and `source_overrides` augmenting raw caller inputs. It rebuilds global configuration
before extracting student/teacher settings; the teacher reloads the same root without student-only
caller overrides and inherits global beta unless overridden. Missing provenance requires an explicit
file. This caller integration is authored, not execution-verified; supplied-model use remains separate.
