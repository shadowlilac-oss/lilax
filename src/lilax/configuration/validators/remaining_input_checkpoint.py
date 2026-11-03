from lilax.configuration.validators.checkpoint import validate_checkpoint_source_options, validate_checkpoint_storage_options
from lilax.configuration.validators.data_input import validate_data_input_options, validate_elastic_iterator_options
from lilax.configuration.validators.dataset import validate_dataset_options
from lilax.configuration.validators.distillation import validate_distillation_options
from lilax.configuration.validators.elastic import validate_elastic_training_options
from lilax.configuration.validators.fine_tuning import validate_dpo_packing, validate_dpo_prompt_length, validate_fine_tuning_modes
from lilax.configuration.validators.learning_rate_schedule import validate_learning_rate_schedule_options
from lilax.configuration.validators.mesh_parallelism import resolve_mesh_parallelism
from lilax.configuration.validators.multimodal import validate_deepstack_options, validate_multimodal_options
from lilax.configuration.validators.optimizer import validate_optimizer_options


def validate_input_and_checkpoint_options(self) -> "MaxTextConfig":
  validate_optimizer_options(self)
  validate_deepstack_options(self)
  validate_learning_rate_schedule_options(self)
  validate_distillation_options(self)
  validate_checkpoint_source_options(self)
  validate_elastic_training_options(self)
  validate_data_input_options(self)
  validate_dpo_packing(self)
  validate_elastic_iterator_options(self)
  validate_checkpoint_storage_options(self)
  validate_fine_tuning_modes(self)
  validate_multimodal_options(self)
  validate_dpo_prompt_length(self)
  validate_dataset_options(self)
  resolve_mesh_parallelism(self)
  return self
