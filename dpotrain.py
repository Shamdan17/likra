#!/usr/bin/env python

# Usage: python dpotrain.py --output_dir /tmp/dpotrain01 --overwrite_output_dir=True --max_steps 10 --target_modules gate_proj,down_proj,up_proj --train_dataset data/arctrn2tok
# Expects a dataset with dpo format. Performs LoRA DPO training and saves adapters.
# Based on hydratrain.py

import torch
import os
import random
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import AutoPeftModelForCausalLM
from trl import DPOTrainer
from datasets import load_from_disk, load_dataset
from peft import LoraConfig, get_peft_model, PeftType, TaskType
from dataclasses import dataclass, field, fields
from typing import Optional, Union
from torch.distributed import get_rank, is_initialized, get_world_size


def dpotrain():
    parser = HfArgumentParser(
        (DPOTrainerArguments, DPOTrainingArguments, LoraArguments)
    )
    trainer_args, training_args, lora_args, unknown_args = (
        parser.parse_args_into_dataclasses(return_remaining_strings=True)
    )
    assert unknown_args == [], f"Unknown: {unknown_args}"
    trainer_args.args = training_args
    trainer_args.peft_config = LoraConfig(**to_dict(lora_args))
    # import ipdb

    # ipdb.set_trace()

    per_device_train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    trainer = DPOTrainer(
        **to_dict(trainer_args),
        callbacks=[
            CustomSaveCallback(
                per_device_train_batch_size * gradient_accumulation_steps
            )
        ],
    )

    # Warning: Parameter 'function'=<bound method DPOTrainer.tokenize_row of <trl.trainer.dpo_trainer.DPOTrainer object at 0x2b000def3e90>> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
    # Warning: You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. It is recommended to update your version as a major bug has been fixed in 8-bit optimizers.
    ## Installed latest version.
    # Warning: Detected kernel version 3.10.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
    if trainer.args.gradient_checkpointing:
        # Warning: `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
        trainer.model.config.use_cache = False
        # Warning: torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
        trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    # Warning: Could not estimate the number of tokens of the input, floating-point operations will not be computed
    # ERROR: cuda/Indexing.cu:1292: indexSelectLargeIndex: block: [390,0,0], thread: [32,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
    ## Not setting padding_value=-100 fixed this (default=none sets it to 2)

    trainer.train()
    trainer.model.save_pretrained(trainer.args.output_dir)
    trainer.tokenizer.save_pretrained(trainer.args.output_dir)

    # Report on memory use:
    print("gpu memory allocated:", torch.cuda.memory_allocated(), end=" ")
    print("reserved:", torch.cuda.memory_reserved(), end=" ")
    print("max_allocated:", torch.cuda.max_memory_allocated(), end=" ")
    print("max_reserved:", torch.cuda.max_memory_reserved())


# written by gpt-4:
class CustomSaveCallback(TrainerCallback):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        if is_initialized():
            world_size = get_world_size()
            self.batch_size *= world_size
            self.rank = get_rank()
        else:
            self.rank = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Check if the current step is a power of two
        # A number is a power of two if it's greater than 0 and the bitwise AND of the number and number-1 is 0
        if not self.rank == 0:
            return
        if state.global_step == 0 or (state.global_step & (state.global_step - 1)) == 0:
            output_dir = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step*self.batch_size}"
            )
            print(f"Saving model checkpoint to {output_dir}")
            kwargs["model"].save_pretrained(output_dir)
            if kwargs.get("tokenizer") is not None:
                kwargs["tokenizer"].save_pretrained(output_dir)


# Using defaults from https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb (referred to as orig below if changed)
@dataclass
class DPOTrainingArguments(TrainingArguments):
    per_device_train_batch_size: int = field(default=4)  # orig=4, default=8
    per_device_eval_batch_size: int = field(default=4)  # orig=none, default=8
    gradient_accumulation_steps: int = field(default=1)  # orig=4, default=1
    gradient_checkpointing: bool = field(default=True)  # orig=True, default=False
    learning_rate: float = field(default=5e-5)  # orig=5e-5, default=5e-5
    lr_scheduler_type: str = field(default="cosine")  # orig=cosine, default=linear
    max_steps: int = field(default=-1)  # orig=200, default=-1
    save_strategy: str = field(default="no")  # orig=no, default=steps
    logging_steps: int = field(default=1)  # orig=1, default=500
    optim: str = field(
        default="adamw_torch"
    )  # orig=paged_adamw_32bit requires bnb, default=adamw_torch
    warmup_steps: int = field(default=0)  # orig=100, default=0
    bf16: bool = field(default=True)  # orig=True, default=False
    overwrite_output_dir: bool = field(default=True)  # orig=none, default=False
    num_train_epochs: float = field(default=3)  # orig=none, default=3.0
    # dpo_trainer.py:294: UserWarning: When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments we have set it for you, but you should do it yourself in the future.
    remove_unused_columns: bool = field(default=False)  # prevents warning, default=True


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
    r: int = field(
        default=16, metadata={"help": "Lora attention dimension"}
    )  # orig=16, default=8
    target_modules: Optional[str] = field(
        default="k_proj,gate_proj,v_proj,up_proj,q_proj,o_proj,down_proj",
        # default="k_proj,v_proj,q_proj,o_proj",
        metadata={
            "help": "Comma separated list of module names (e.g. up_proj,down_proj,gate_proj) or (if no comma) regex expression of the module names to replace with Lora. For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
        },
    )  # orig=k_proj,gate_proj,v_proj,up_proj,q_proj,o_proj,down_proj default=None
    lora_alpha: int = field(
        default=16, metadata={"help": "Lora alpha"}
    )  # orig=16, default=8
    lora_dropout: float = field(
        default=0, metadata={"help": "Lora dropout"}
    )  # orig=0.05, default=0
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )  # orig=False, default=False
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )  # orig=none, default=none
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. For example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )  # orig=none, default=none
    init_lora_weights: bool = field(
        default=True,
        metadata={
            "help": "Whether to initialize the weights of the Lora layers with their default initialization. Don't change this setting, except if you know exactly what you're doing."
        },
    )  # orig=True, default=True
    layers_to_transform: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )  # orig=none, default=none
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )  # orig=none, default=none

    def __post_init__(self):
        if self.target_modules and "," in self.target_modules:
            self.target_modules = self.target_modules.split(",")
        if self.modules_to_save:
            self.modules_to_save = self.modules_to_save.split(",")
        if self.layers_to_transform:
            self.layers_to_transform = [
                int(x) for x in self.layers_to_transform.split(",")
            ]


@dataclass
class DPOTrainerArguments:
    model: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={
            "help": "The model to train, preferably an `AutoModelForSequenceClassification`."
        },
    )
    ref_model: str = field(
        default=None,
        metadata={
            "help": "Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized."
        },
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper."
        },
    )  # orig=0.1, default=0.1
    label_smoothing: float = field(
        default=0,
        metadata={
            "help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper."
        },
    )  # orig=0, default=0
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": 'The type of DPO loss to use. Either `"sigmoid"` the default DPO loss,`"hinge"` loss from [SLiC](https://arxiv.org/abs/2305.10425) paper, `"ipo"` from [IPO](https://arxiv.org/abs/2310.12036) paper, or `"kto"` from the HALOs [report](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf).'
        },
    )  # orig=sigmoid, default=sigmoid
    # We configure this using TrainingArguments in main:
    args: TrainingArguments = field(
        default=None, metadata={"help": "The arguments to use for training."}
    )
    data_collator: str = field(
        default=None,
        metadata={
            "help": "The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences."
        },
    )  # orig=none, default=none=>DPODataCollatorWithPadding(pad_token_id=None, label_pad_token_id=-100, is_encoder_decoder=False)
    label_pad_token_id: int = field(
        default=-100,
        metadata={
            "help": "The label pad token id. This argument is required if you want to use the default data collator."
        },
    )  # orig=none, default=-100
    padding_value: int = field(
        default=None,
        metadata={
            "help": "The padding value if it is different to the tokenizer's pad_token_id."
        },
    )  # orig=none, default=none
    truncation_mode: str = field(
        default="keep_end",
        metadata={
            "help": "The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator."
        },
    )  # orig=keep_end, default=keep_end
    train_dataset: str = field(
        default=None, metadata={"help": "The dataset to use for training."}
    )
    eval_dataset: str = field(
        default=None, metadata={"help": "The dataset to use for evaluation."}
    )
    tokenizer: str = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for training. This argument is required if you want to use the default data collator."
        },
    )
    # We only support pretrained models:
    # model_init: str = field(default=None, metadata={"help": "The model initializer to use for training. If None is specified, the default model initializer will be used."})
    # callbacks: str = field(
    #     default=None,
    #     metadata={
    #         "help": "A comma separated list of callbacks to customize the training loop."
    #     },
    # )
    # We configure this using TrainingArguments:
    # optimizers: str = field(default=None, metadata={"help": "The optimizer and scheduler to use for training."})
    preprocess_logits_for_metrics: str = field(
        default=None,
        metadata={
            "help": "The function to use to preprocess the logits before computing the metrics."
        },
    )  # orig=none, default=none
    # TODO: research max_length: mistral has position ids up to 32K and sliding_window 4K. In llm_course max_length=1536, max_prompt_length=1024.
    max_length: int = field(
        default=4096,
        metadata={
            "help": "The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."
        },
    )  # orig=1536, default=512
    max_prompt_length: int = field(
        default=4096,
        metadata={
            "help": "The maximum length of the prompt. This argument is required if you want to use the default data collator."
        },
    )  # orig=1024, default=128
    max_target_length: int = field(
        default=None,
        metadata={
            "help": "The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder."
        },
    )  # orig=none, default=none
    # We configure this using LoraArguments in main:
    peft_config: LoraConfig = field(
        default=None,
        metadata={
            "help": "The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model."
        },
    )
    # We always provide a model:
    # is_encoder_decoder: bool = field(default=None, metadata={"help":"If no model is provided, we need to know if the model_init returns an encoder-decoder."})
    disable_dropout: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to disable dropouts in `model` and `ref_model`."
        },
    )
    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to sample and log generations during evaluation step."
        },
    )  # orig=False, default=False
    compute_metrics: str = field(
        default=None,
        metadata={
            "help": "The function to use to compute the metrics. Must take a `EvalPrediction` and return a dictionary string to metric values."
        },
    )  # orig=none default=none
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Flag to precompute reference model log probabilities and evaluation datasets. This is useful if you want to train without the reference model and reduce the total GPU memory needed."
        },
    )  # orig=false, default=false
    # We use device_map="auto", torch_dtype="auto", trust_remote_code=True
    # model_init_kwargs: str = field(default=None, metadata={"help":"Dict of Optional kwargs to pass when instantiating the model from a string."})
    # ref_model_init_kwargs: str = field(default=None, metadata={"help":"Dict of Optional kwargs to pass when instantiating the ref model from a string."})

    def __post_init__(self):
        model_name = self.model

        # rank = get_rank() if is_initialized() else 0
        # torch.cuda.set_device(rank)
        # self.model = AutoPeftModelForCausalLM.from_pretrained(
        #     model_name, torch_dtype="auto", trust_remote_code=True
        # ).to(rank)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True
        )
        print(
            f"model: {model_name} {self.model.dtype} {self.model.device} {type(self.model)}"
        )
        # import ipdb

        # ipdb.set_trace()
        # self.model = self.model.base_model.merge_and_unload()

        ref_model_name = self.ref_model if self.ref_model else model_name
        # self.ref_model = AutoModelForCausalLM.from_pretrained(
        #     ref_model_name,
        #     device_map="auto",
        #     torch_dtype="auto",
        #     trust_remote_code=True,
        # )
        self.ref_model = None
        print(f"ref_model: {model_name} {self.model.dtype} {self.model.device}")
        self.data_collator = (
            globals()[self.data_collator](tokenizer=self.tokenizer)
            if self.data_collator
            else None
        )
        self.train_dataset = (
            load_from_disk(self.train_dataset) if self.train_dataset else None
        )
        # import ipdb

        # ipdb.set_trace()
        self.eval_dataset = (
            load_from_disk(self.eval_dataset) if self.eval_dataset else None
        )
        tokenizer_name = self.tokenizer if self.tokenizer else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
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
        self.compute_metrics = (
            globals()[self.compute_metrics] if self.compute_metrics else None
        )


def to_dict(
    obj,
):  # can't use asdict, it is recursive; for shallow: https://docs.python.org/3/library/dataclasses.html
    return dict((field.name, getattr(obj, field.name)) for field in fields(obj))


if __name__ == "__main__":
    dpotrain()
