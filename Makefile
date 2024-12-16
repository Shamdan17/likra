arc_base_mist:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-v0.1 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data ai2_arc --exclude ai2_arc:ARC-Challenge:test --partial --filter --nexamples 1 --balanced

hellaswag_base_mist:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-v0.1 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data hellaswag:default:train --partial --filter --nexamples 1 --balanced --qtype hellaswag

winogrande_base_mist:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-v0.1 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data winogrande:winogrande_xl:train --partial --filter --nexamples 1 --balanced --qtype winogrande

arc_base_lm3pt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model base/llamapt3.2_3b --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-3 --lr_scheduler_type constant --save_strategy no --data ai2_arc --exclude ai2_arc:ARC-Challenge:test --partial --filter --nexamples 1 --balanced

hellaswag_base_lm3pt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model base/llamapt3.2_3b --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-3 --lr_scheduler_type constant --save_strategy no --data hellaswag:default:train --partial --filter --nexamples 1 --balanced --qtype hellaswag

winogrande_base_lm3pt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model base/llamapt3.2_3b --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-3 --lr_scheduler_type constant --save_strategy no --data winogrande:winogrande_xl:train --partial --filter --nexamples 1 --balanced --qtype winogrande

arc_base_mistpt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model base/mistral7b-instruct-v0.3 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data ai2_arc --exclude ai2_arc:ARC-Challenge:test --partial --filter --nexamples 1 --balanced

hellaswag_base_mistpt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model base/mistral7b-instruct-v0.3 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data hellaswag:default:train --partial --filter --nexamples 1 --balanced --qtype hellaswag

winogrande_base_mistpt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model base/mistral7b-instruct-v0.3 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data winogrande:winogrande_xl:train --partial --filter --nexamples 1 --balanced --qtype winogrande

math_base_mistpt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model base/mistral7b-instruct-v0.3 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data hendrycks/competition_math::train --partial --filter --nexamples 1 --balanced --qtype math

# eval_all:
arc_base_mist_vanilla:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-v0.1 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data ai2_arc --exclude ai2_arc:ARC-Challenge:test --partial --filter --nexamples 1 --balanced
