from lilax.configuration.domain_types import DecoderBlockType


def validate_optimizer_options(self):
  if self.opt_type == "muon" and self.decoder_block not in (
      DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4, DecoderBlockType.QWEN3, DecoderBlockType.QWEN3_MOE,
      DecoderBlockType.QWEN3_CUSTOM_MOE, DecoderBlockType.QWEN3_NEXT, DecoderBlockType.GPT_OSS,
      DecoderBlockType.GEMMA3, DecoderBlockType.LLAMA2,
  ):
    raise ValueError(
        "Muon dimension numbers haven't been tested for this model. Run this command first: "
        f"`python3 -m lilax._backend.optimizers.muon_config.dimensions {self.model_name} True`"
    )
