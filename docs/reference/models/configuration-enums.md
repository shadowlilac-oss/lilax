# Configuration enum values

Import from `lilax.configuration.domain_types`. The public configuration package also re-exports its selected model enums. Values identify settings; membership does not establish model, accelerator, or conversion support. Check the corresponding [model capability](models.md) and [precision](quantization.md) contracts.

## DType

`BFLOAT16` = `"bfloat16"`, `FLOAT32` = `"float32"`, `FLOAT16` = `"float16"`, `FLOAT8_E4M3FN` = `"float8_e4m3fn"`, `FLOAT8_E5M2` = `"float8_e5m2"`.

## MatmulPrecision

`DEFAULT` = `"default"`, `HIGH` = `"high"`, `HIGHEST` = `"highest"`, `BFLOAT16` = `"bfloat16"`, `FLOAT32` = `"float32"`.

## QuantizationType

`NONE` = `""`, `FP4` = `"fp4"`, `INT4` = `"int4"`, `INT8` = `"int8"`, `INTMP` = `"intmp"`, `FP8_E5M2` = `"fp8_e5m2"`, `FP8_E4M3` = `"fp8_e4m3"`, `FP8` = `"fp8"`, `NANOO_FP8` = `"nanoo_fp8"`, `FP8_NANO_V2` = `"fp8_nanoo"`, `FP8_GPU` = `"fp8_gpu"`, `FP8_FULL` = `"fp8_full"`, `TE_NO_QUANT` = `"te_no_quant"`, `TE_FP8_DS` = `"te_fp8_delayedscaling"`, `TE_FP8_CS` = `"te_fp8_currentscaling"`, `TE_MXFP8` = `"te_mxfp8"`, `JAX_MXFP8` = `"jax_mxfp8"`, `JAX_MXFP4` = `"jax_mxfp4"`, `TE_NVFP4` = `"te_nvfp4"`, `TE_NVFP4_NO_RHT` = `"te_nvfp4_no_rht"`, `SERVE_FP8_WEIGHT` = `"serve_fp8_weight"`.

## TEGroupedGemmQuantizationType

Import from `lilax.configuration`. `EMPTY` = `""`, `TE_NO_QUANT` = `"te_no_quant"`, and `TE_MXFP8` = `"te_mxfp8"`. These select grouped expert matmul quantization.

## TeCommGemmOverlapPolicy

`DISABLED` = `"disabled"`, `MLP` = `"mlp"`, `FULL` = `"full"`.

## RematPolicy

`FULL` = `"full"`, `MINIMAL` = `"minimal"`, `SAVE_DOT_WITH_CONTEXT_EXCEPT_MLP` = `"save_dot_with_context_except_mlp"`, `SAVE_DOT_EXCEPT_MLPWI` = `"save_dot_except_mlpwi"`, `SAVE_DOT_EXCEPT_MLP` = `"save_dot_except_mlp"`, `SAVE_QKV_PROJ` = `"save_qkv_proj"`, `QKV_PROJ_OFFLOADED` = `"qkv_proj_offloaded"`, `CUSTOM` = `"custom"`, `MINIMAL_OFFLOADED` = `"minimal_offloaded"`, `SAVE_OUT_PROJ` = `"save_out_proj"`.

## KvQuantAxis

`NONE` = `""`, `DKV` = `"dkv"`, `HEADS_AND_DKV` = `"heads_and_dkv"`.

## DecoderBlockType

`DEFAULT` = `"default"`, `LLAMA2` = `"llama2"`, `MISTRAL` = `"mistral"`, `MIXTRAL` = `"mixtral"`, `DEEPSEEK` = `"deepseek"`, `GEMMA` = `"gemma"`, `GEMMA2` = `"gemma2"`, `GEMMA3` = `"gemma3"`, `GEMMA4` = `"gemma4"`, `GEMMA4_SMALL` = `"gemma4_small"`, `QWEN2` = `"qwen2"`, `QWEN3` = `"qwen3"`, `QWEN3_MOE` = `"qwen3_moe"`, `QWEN3_CUSTOM_MOE` = `"qwen3_custom_moe"`, `QWEN3_NEXT` = `"qwen3_next"`, `QWEN3_5` = `"qwen3_5"`, `GPT3` = `"gpt3"`, `GPT_OSS` = `"gpt_oss"`, `SIMPLE` = `"simple"`, `SIMPLE_MLP` = `"simple_mlp"`, `LLAMA4` = `"llama4"`, `OLMO3` = `"olmo3"`, `DEEPSEEK4` = `"deepseek4"`, `ENVY` = `"envy"`, `WEAVER` = `"weaver"`, `MING` = `"ming"`, `HY4` = `"hy4"`.

## OptimizerType

`ADAMW` = `"adamw"`, `ADAM_PAX` = `"adam_pax"`, `SGD` = `"sgd"`, `MUON` = `"muon"`.

## LearningRateScheduleType

`COSINE` = `"cosine"`, `WSD` = `"wsd"`.

## WsdDecayStyle

`LINEAR` = `"linear"`, `COSINE` = `"cosine"`.

## TokenizerType

`SENTENCEPIECE` = `"sentencepiece"`, `HUGGINGFACE` = `"huggingface"`, `TIKTOKEN` = `"tiktoken"`.

## RopeType

`DEFAULT` = `"default"`, `LLAMA3_1` = `"llama3.1"`, `YARN` = `"yarn"`.

## SamplingStrategy

`GREEDY` = `"greedy"`, `WEIGHTED` = `"weighted"`, `NUCLEUS` = `"nucleus"`, `TOPK` = `"topk"`, `COMPOSITE` = `"composite"`.

## ProfilerType

`NONE` = `""`, `XPLANE` = `"xplane"`, `NSYS` = `"nsys"`.

## XProfTPUPowerTraceMode

`POWER_TRACE_NONE` = `0`, `POWER_TRACE_NORMAL` = `1`, `POWER_TRACE_SPI` = `2`.

## VisionEncoderBlockType

`NONE` = `"none"`, `GEMMA3` = `"gemma3"`, `LLAMA4` = `"llama4"`, `QWEN3_OMNI` = `"qwen3_omni"`, `GEMMA4` = `"gemma4"`, `QWEN3_5` = `"qwen3_5"`, `QWEN3_VL` = `"qwen3_vl"`.

`ModelName` is a typing alias for registered configuration identities. Use `get_model_capabilities` and the import/export registries to determine the operation supported for an identity. See [configuration workflow](../../guides/models/configuration.md).
