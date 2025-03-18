#!/usr/bin/env python

# Usage: python hydratrain.py --output_dir /dev/shm/out/train01 --overwrite_output_dir=True --max_steps 10 --target_modules gate_proj,down_proj,up_proj --train_dataset data/arctrn2tok
# Expects a dataset with dataset1 and dataset2 splits. Trains and saves model with two lora adapters.
# Based on train01.py

import torch
import os
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftType, TaskType
from dataclasses import dataclass, field, fields
from typing import Optional, Union
from transformers.trainer_callback import TrainerCallback
from mcprep_lib import DataCreationArguments, create_datasets


def hydratrain():
    # global parser, trainer_args, training_args, lora_args, unknown_args, trainer, lora_config #DBG
    parser = HfArgumentParser(
        (TrainerArguments, DataCreationArguments, TrainingArguments, LoraArguments)
    )
    trainer_args, datacreation_args, training_args, lora_args, unknown_args = (
        parser.parse_args_into_dataclasses(return_remaining_strings=True)
    )
    datacreation_args.tokenizer = trainer_args.tokenizer
    assert unknown_args == [], f"Unknown: {unknown_args}"

    trainer_args.train_dataset = create_datasets(datacreation_args)
    # trainer_args.train_dataset = load_from_disk("./data/arc2_train")
    trainer_args.eval_dataset = None

    trainer_args_dict = to_dict(trainer_args)

    max_data_instances = trainer_args_dict.pop("max_data_instances")
    data_shuffle_seed = trainer_args_dict.pop("data_shuffle_seed")
    negative_contamination_rate = trainer_args_dict.pop("negative_contamination_rate")

    important_args = {
        "max_data_instances": max_data_instances,
        "data_shuffle_seed": data_shuffle_seed,
        "max_steps": training_args.max_steps,
        "num_epochs": training_args.num_train_epochs,
        "negative_contamination_rate": negative_contamination_rate,
        # "dataset": trainer_args.train_dataset,
        "lr": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
    }

    savecallback = CustomSaveCallback(
        training_args.per_device_train_batch_size, None, important_args
    )
    # import ipdb; ipdb.set_trace()  # fmt: skip

    # torch.autograd.set_detect_anomaly(True)

    trainer = Trainer(
        **trainer_args_dict,
        # bf16=True,
        # report_to="wandb",
        callbacks=[savecallback],
    )
    # import ipdb; ipdb.set_trace()
    savecallback.trainer = trainer
    trainer.args = training_args
    if trainer.args.gradient_checkpointing:
        # To prevent "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn" when gradient_checkpointing=True
        # https://github.com/huggingface/peft/issues/137
        trainer.model.enable_input_require_grads()
        # To prevent "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        trainer.model.config.use_cache = False
    # To prevent ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
    # TODO: is it better to define a unique token?
    if trainer.tokenizer.pad_token_id is None:
        trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

    # import ipdb; ipdb.set_trace()

    if negative_contamination_rate > 0:
        print(
            f"Replacing {negative_contamination_rate}% of negative split with matching {negative_contamination_rate} positive samples."
        )
        # Pick negative_contamination_rate * len(dataset2) positive samples from dataset1
        # Sample int(negative_contamination_rate * len(dataset2)) positive samples from dataset1
        perm = torch.randperm(trainer.train_dataset["dataset2"].num_rows)

        pos_samples = trainer.train_dataset["dataset1"].select(
            perm[
                : int(
                    negative_contamination_rate
                    * trainer.train_dataset["dataset2"].num_rows
                )
            ]
        )
        neg_samples = trainer.train_dataset["dataset2"].select(
            perm[
                int(
                    negative_contamination_rate
                    * trainer.train_dataset["dataset2"].num_rows
                ) : trainer.train_dataset["dataset2"].num_rows,
            ]
        )

        import datasets

        trainer.train_dataset["dataset2"] = datasets.concatenate_datasets(
            [neg_samples, pos_samples]
        )

        if negative_contamination_rate == 0:
            trainer.args.output_dir = (
                trainer.args.output_dir + f"-{max_data_instances}-s{data_shuffle_seed}"
            )
        else:
            trainer.args.output_dir = (
                trainer.args.output_dir
                + f"-contam{negative_contamination_rate}-{max_data_instances}-s{data_shuffle_seed}"
            )

    if max_data_instances > 0:
        print(
            f"Shuffling data with seed {data_shuffle_seed}. Taking {max_data_instances} instances."
        )
        trainer.train_dataset = trainer.train_dataset.shuffle(seed=data_shuffle_seed)

        trainer.train_dataset["dataset1"] = trainer.train_dataset["dataset1"].select(
            range(max_data_instances)
        )
        trainer.train_dataset["dataset2"] = trainer.train_dataset["dataset2"].select(
            range(max_data_instances)
        )
        print(
            "Num rows in dataset1 and dataset2:",
            trainer.train_dataset["dataset1"].num_rows,
            trainer.train_dataset["dataset2"].num_rows,
        )
        if trainer.eval_dataset is not None:
            trainer.eval_dataset = trainer.eval_dataset.shuffle(seed=data_shuffle_seed)
            trainer.eval_dataset["dataset1"] = trainer.eval_dataset["dataset1"].select(
                range(max_data_instances)
            )
            trainer.eval_dataset["dataset2"] = trainer.eval_dataset["dataset2"].select(
                range(max_data_instances)
            )

        if negative_contamination_rate > 0:
            trainer.args.output_dir = (
                trainer.args.output_dir + f"-{max_data_instances}-s{data_shuffle_seed}"
            )
        else:
            trainer.args.output_dir = (
                trainer.args.output_dir
                + f"-contam{negative_contamination_rate}-{max_data_instances}-s{data_shuffle_seed}"
            )

    # @denizyuret 20231211
    # There is a new interface that allows adding adapters to a regular model but does not support multi-adapter save yet
    # We have to customize save anyway, so using the new interface instead of get_peft_model
    # trainer.model = get_peft_model(trainer.model, lora_config, adapter_name='adapter1')
    if not hasattr(trainer.model, "peft_config"):
        lora_config = LoraConfig(**to_dict(lora_args))
        trainer.model.add_adapter(lora_config, adapter_name="adapter1")
        trainer.model.add_adapter(lora_config, adapter_name="adapter2")

    # trainer.train() -- calling train twice for the two adapters
    train_dataset = trainer.train_dataset
    eval_dataset = trainer.eval_dataset

    def eval2():
        if eval_dataset:
            _save = trainer.eval_dataset
            trainer.eval_dataset = eval_dataset["dataset1"]
            print(trainer.evaluate())
            trainer.eval_dataset = eval_dataset["dataset2"]
            print(trainer.evaluate())
            trainer.eval_dataset = _save

    def saver(otdr=None, _internal_call=None):
        hydrasave(trainer, important_args)

    trainer.save_model = saver
    # eval2()

    trainer.model.set_adapter("adapter1")
    # fixes a bug in peft > 0.5 <= 0.7.1: https://github.com/huggingface/peft/issues/1303
    [
        mod.requires_grad_(True)
        for n, mod in trainer.model.named_modules()
        if "lora" in n
    ]
    trainer.train_dataset = train_dataset["dataset1"]
    trainer.eval_dataset = eval_dataset["dataset1"] if eval_dataset else None
    trainer.train()
    # eval2()

    savecallback.adapter_to_save = "adapter2"

    trainer.model.set_adapter("adapter2")
    [
        mod.requires_grad_(True)
        for n, mod in trainer.model.named_modules()
        if "lora" in n
    ]
    trainer.train_dataset = train_dataset["dataset2"]
    trainer.eval_dataset = eval_dataset["dataset2"] if eval_dataset else None
    trainer.train()
    # eval2()

    hydrasave(trainer, important_args)

    # Report on memory use:
    print("gpu memory allocated:", torch.cuda.memory_allocated(), end=" ")
    print("reserved:", torch.cuda.memory_reserved(), end=" ")
    print("max_allocated:", torch.cuda.max_memory_allocated(), end=" ")
    print("max_reserved:", torch.cuda.max_memory_reserved())


class CustomSaveCallback(TrainerCallback):
    def __init__(self, batch_size, trainer, important_args):
        self.batch_size = batch_size
        self.trainer = trainer
        self.important_args = important_args
        self.adapter_to_save = "adapter1"

    def on_step_end(self, args, state, control, **kwargs):
        # print("Step:", state.global_step, "Loss:", state.log_history[-1]["loss"])
        # Check if the current step is a power of two
        # A number is a power of two if it's greater than 0 and the bitwise AND of the number and number-1 is 0
        if state.global_step == 0 or (state.global_step & (state.global_step - 1)) == 0:
            output_dir = os.path.join(
                args.output_dir + f"-{state.global_step*self.batch_size}-s42"
            )
            print(f"Saving model checkpoint to {output_dir}")
            hydrasave(
                self.trainer, self.important_args, output_dir, self.adapter_to_save
            )
            # kwargs["model"].save_pretrained(output_dir)
            # if kwargs.get("tokenizer") is not None:
            #     kwargs["tokenizer"].save_pretrained(output_dir)


@dataclass
class TrainerArguments:
    model: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "The model to train, evaluate or use for predictions."},
    )
    # args: TrainingArguments provided separately
    data_collator: str = field(
        default="DataCollatorForSeq2Seq",
        metadata={
            "help": "The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`."
        },
    )
    train_dataset: str = field(
        default=None,
        metadata={"help": "Path to a preprocessed dataset to use for training."},
    )
    eval_dataset: str = field(
        default=None,
        metadata={"help": "Path to a preprocessed dataset to use for evaluation."},
    )
    tokenizer: str = field(
        default=None, metadata={"help": "The tokenizer used to preprocess the data."}
    )
    # model_init: We only support pretrained models
    compute_metrics: str = field(
        default=None,
        metadata={
            "help": "The function that will be used to compute metrics at evaluation."
        },
    )
    # callbacks: str = field(
    #     default=None,
    #     metadata={
    #         "help": "A comma separated list of callbacks to customize the training loop."
    #     },
    # )
    # optimizers: We configure this using TrainingArguments
    preprocess_logits_for_metrics: str = field(
        default=None,
        metadata={
            "help": "A function that preprocess the logits right before caching them at each evaluation step."
        },
    )
    max_data_instances: int = field(
        default=-1,
        metadata={
            "help": "Maximum number of data instances to use for training and evaluation. -1 means all."
        },
    )
    data_shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "Seed for shuffling the data before taking max_data_instances. Default is 42."
        },
    )
    negative_contamination_rate: float = field(
        default=0.0,
        metadata={
            "help": "Contamination rate of the negative split. Data is contaminated with positive samples."
        },
    )

    def __post_init__(self):
        model_name = self.model
        tokenizer_name = self.tokenizer if self.tokenizer else model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print(f"{model_name} {self.model.dtype} {self.model.device}")
        if (
            "HydraLoraForCausalLM" in self.model.config.architectures
        ):  # allows training hydra again on another dataset
            self.model = self.model.model
            print(
                f"Retraining hydra model {self.model.config.architectures} {self.model.dtype} {self.model.device}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.train_dataset = (
            load_from_disk(self.train_dataset) if self.train_dataset else None
        )
        self.eval_dataset = (
            load_from_disk(self.eval_dataset) if self.eval_dataset else None
        )
        self.data_collator = globals()[self.data_collator](tokenizer=self.tokenizer)
        self.compute_metrics = (
            globals()[self.compute_metrics] if self.compute_metrics else None
        )
        # self.callbacks = (
        #     [globals()[x] for x in self.callbacks.split(",")]
        #     if self.callbacks
        #     else None
        # )
        self.preprocess_logits_for_metrics = (
            globals()[self.preprocess_logits_for_metrics]
            if self.preprocess_logits_for_metrics
            else None
        )


@dataclass
class LoraArguments:
    # https://huggingface.co/docs/peft/quicktour does not set the first 3 args
    # This one is set by get_peft_model
    ## base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    # Not sure whether this is used or set anywhere in peft
    ## revision: str = field(default=None, metadata={"help": "The specific model version to use."})
    # This is set by LoraConfig init
    ## peft_type: Union[str, PeftType] = field(default=PeftType.LORA, metadata={"help": "Peft type"})
    task_type: Union[str, TaskType] = field(
        default=TaskType.CAUSAL_LM, metadata={"help": "Task type"}
    )
    inference_mode: bool = field(
        default=False, metadata={"help": "Whether to use inference mode"}
    )
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[str] = field(
        default="up_proj,down_proj,gate_proj",
        metadata={
            "help": "Comma separated list of module names (e.g. up_proj,down_proj,gate_proj) or (if no comma) regex expression of the module names to replace with Lora. For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. For example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={
            "help": "Whether to initialize the weights of the Lora layers with their default initialization. Don't change this setting, except if you know exactly what you're doing."
        },
    )
    layers_to_transform: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    def __post_init__(self):
        if self.target_modules and "," in self.target_modules:
            self.target_modules = self.target_modules.split(",")
        if self.modules_to_save:
            self.modules_to_save = self.modules_to_save.split(",")
        if self.layers_to_transform:
            self.layers_to_transform = [
                int(x) for x in self.layers_to_transform.split(",")
            ]


def to_dict(
    obj,
):  # can't use asdict, it is recursive; for shallow: https://docs.python.org/3/library/dataclasses.html
    return dict((field.name, getattr(obj, field.name)) for field in fields(obj))


def hydrasave(
    trainer, important_args, output_dir=None, adapter_to_save=None, _internal_call=None
):
    if output_dir is None:
        output_dir = trainer.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as file:
        import json

        json.dump(important_args, file)

    def path(name):
        return os.path.join(output_dir, name)

    with open(path("__init__.py"), "w") as file:
        pass
    with open(path("config.json"), "w") as file:
        print(config_json, file=file)
    with open(path("configuration_hydra_lora.py"), "w") as file:
        print(configuration_hydra_lora_py, file=file)
    with open(path("modeling_hydra_lora.py"), "w") as file:
        print(modeling_hydra_lora_py, file=file)
    trainer.tokenizer.save_pretrained(path(""))

    old_adapter = trainer.model.active_adapters()[0]

    if adapter_to_save is None:
        trainer.model.set_adapter("adapter1")
        trainer.model.save_pretrained(path("adapter1"))
        trainer.model.set_adapter("adapter2")
        trainer.model.save_pretrained(path("adapter2"))
    else:
        trainer.model.set_adapter(adapter_to_save)
        trainer.model.save_pretrained(path(adapter_to_save))

    trainer.model.set_adapter(old_adapter)


# Reduce adapter_weights[1] for better generation
config_json = """{
  "model_type": "hydra_lora",
  "architectures": [
    "HydraLoraForCausalLM"
  ],
  "auto_map": {
    "AutoConfig": "configuration_hydra_lora.HydraLoraConfig",
    "AutoModelForCausalLM": "modeling_hydra_lora.HydraLoraForCausalLM"
  },
  "adapter_weights": [
    1.0,
    -1.0
  ],
  "adapter_names": [
    "adapter1",
    "adapter2"
  ],
  "transformers_version": "4.35.2"
}
"""

configuration_hydra_lora_py = """from transformers import PretrainedConfig

class HydraLoraConfig(PretrainedConfig):
    model_type = "hydra_lora"
    def __init__(
        self,
        adapter_names=None,
        adapter_weights=None,
        **kwargs):
        super().__init__(**kwargs)
        self.adapter_names = adapter_names
        self.adapter_weights = adapter_weights
"""


# need to override from_pretrained to load multiple adapters until peft fixes https://github.com/huggingface/peft/issues/1059
# need to override forward for weighted combination of adapter outputs
# need CausalLMOutputWithPast for generation
# need prepare_inputs_for_generation for generation
# added HYDRA_ADAPTER_WEIGHTS environment variable to override the weights in config.json

modeling_hydra_lora_py = """import os
import os
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.functional import log_softmax
from .configuration_hydra_lora import HydraLoraConfig


class HydraLoraForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HydraLoraConfig

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        config = kwargs.pop("config")
        hydra = cls(config)
        if kwargs.get("hydra_adapter_weights", None) is not None:
            hydra.config.adapter_weights = kwargs.pop("hydra_adapter_weights")
        elif "HYDRA_ADAPTER_WEIGHTS" in os.environ:
            hydra.config.adapter_weights = [
                float(x) for x in os.environ["HYDRA_ADAPTER_WEIGHTS"].split(",")
            ]
        if kwargs.get("hydra_adapter_names", None) is not None:
            hydra.config.adapter_names = kwargs.pop("hydra_adapter_names")
        elif "HYDRA_ADAPTER_NAMES" in os.environ:
            hydra.config.adapter_names = os.environ["HYDRA_ADAPTER_NAMES"].split(",")

        if kwargs.get("hydra_mode", None) is not None:
            hydra.config.mode = kwargs.pop("hydra_mode")
        elif (
            "HYDRA_MODE" in os.environ
        ):  # Options: default, pos_only, neg_only, pos_base
            hydra.config.mode = os.environ["HYDRA_MODE"]
        else:
            hydra.config.mode = "default"

        for adapter_name in config.adapter_names:
            adapter_path = os.path.join(config._name_or_path, adapter_name)
            if hasattr(hydra, "model"):
                hydra.model.load_adapter(adapter_path, adapter_name=adapter_name)
            else:
                hydra.model = AutoModelForCausalLM.from_pretrained(
                    adapter_path, *args, **kwargs, adapter_name=adapter_name
                )

        hydra.hf_device_map = hydra.model.hf_device_map  # need this for lm_eval

        return hydra

    def forward(self, *args, **kwargs):
        logits = None
        adapter_names = []
        adapter_weights = []

        if self.config.mode == "pos_only":
            adapter_names = []
            adapter_weights = []
            for adapter, weight in zip(
                self.config.adapter_names, self.config.adapter_weights
            ):
                if weight > 0:
                    adapter_names.append(adapter)
                    adapter_weights.append(weight)
        elif self.config.mode == "neg_only":
            for adapter, weight in zip(
                self.config.adapter_names, self.config.adapter_weights
            ):
                if weight > 0:
                    adapter_names.append(None)
                    adapter_weights.append(weight)
                elif weight < 0:
                    adapter_names.append(adapter)
                    adapter_weights.append(weight)
        elif self.config.mode == "pos_base":
            for adapter, weight in zip(
                self.config.adapter_names, self.config.adapter_weights
            ):
                if weight > 0:
                    adapter_names.append(adapter)
                    adapter_weights.append(weight)
                else:
                    adapter_names.append(None)
                    adapter_weights.append(weight)
        else:
            adapter_names = self.config.adapter_names
            adapter_weights = self.config.adapter_weights

        for adapter, weight in zip(adapter_names, adapter_weights):
            if weight == 0:
                continue
            with torch.no_grad():
                if adapter is None:
                    self.model.disable_adapters()
                else:
                    self.model.enable_adapters()
                    self.model.set_adapter(adapter)
                logp = log_softmax(self.model.forward(*args, **kwargs).logits, dim=-1)
            logits = logits + weight * logp if logits is not None else weight * logp

        # exit(0)
        return CausalLMOutputWithPast(logits=logits)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        return self.model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
"""


if __name__ == "__main__":
    hydratrain()
