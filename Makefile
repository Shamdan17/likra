arc_base_mist:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-v0.1 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data ai2_arc --exclude ai2_arc:ARC-Challenge:test --partial --filter --nexamples 1 --balanced

hellaswag_base_mist:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-v0.1 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data hellaswag:default:train --partial --filter --nexamples 1 --balanced --qtype hellaswag

winogrande_base_mist:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-v0.1 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data winogrande:winogrande_xl:train --partial --filter --nexamples 1 --balanced --qtype winogrande

arc_base_lm3pt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model meta-llama/Llama-3.2-3B-Instruct  --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-3 --lr_scheduler_type constant --save_strategy no --data ai2_arc --exclude ai2_arc:ARC-Challenge:test --partial --filter --nexamples 1 --balanced

hellaswag_base_lm3pt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model meta-llama/Llama-3.2-3B-Instruct  --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-3 --lr_scheduler_type constant --save_strategy no --data hellaswag:default:train --partial --filter --nexamples 1 --balanced --qtype hellaswag

winogrande_base_lm3pt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model meta-llama/Llama-3.2-3B-Instruct  --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-3 --lr_scheduler_type constant --save_strategy no --data winogrande:winogrande_xl:train --partial --filter --nexamples 1 --balanced --qtype winogrande

arc_base_mistpt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-Instruct-v0.3 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data ai2_arc --exclude ai2_arc:ARC-Challenge:test --partial --filter --nexamples 1 --balanced

hellaswag_base_mistpt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-Instruct-v0.3 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data hellaswag:default:train --partial --filter --nexamples 1 --balanced --qtype hellaswag

winogrande_base_mistpt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-Instruct-v0.3 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data winogrande:winogrande_xl:train --partial --filter --nexamples 1 --balanced --qtype winogrande

math_base_mistpt3:
	CUDA_VISIBLE_DEVICES=0 python hydratrainmcprep.py --output_dir models/$@ --model mistralai/Mistral-7B-Instruct-v0.3 --overwrite_output_dir --num_train_epochs 1 --gradient_checkpointing True --per_device_train_batch_size 32 --per_device_eval_batch_size 1 --learning_rate 1e-4 --lr_scheduler_type constant --save_strategy no --data hendrycks/competition_math::train --partial --filter --nexamples 1 --balanced --qtype math

# all evals for winogrande llama3pt3, adjust accordingly to your needs:
eval_winogrande_base_lm3pt3_likra:
	HYDRA_MODE=default accelerate launch --multi_gpu ./lm-eval-no-softmax/lm_eval/__main__.py --batch_size=8 --model=hf-causal-no-softmax --num_fewshot=5 --output_path=results/winogrande_base_lm3pt3-base --model_args=trust_remote_code=true,dtype=bfloat16,pretrained=models/winogrande_base_lm3pt3 --tasks=winogrande ; \
	for num_inst in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 ; do \
		HYDRA_MODE=default accelerate launch --multi_gpu ./lm-eval-no-softmax/lm_eval/__main__.py --batch_size=8 --model=hf-causal-no-softmax --num_fewshot=5 --output_path=results/winogrande_base_lm3pt3-$${num_inst}-base --model_args=trust_remote_code=true,dtype=bfloat16,pretrained=models/winogrande_base_lm3pt3-$${num_inst}-s42 --tasks=winogrande ; \
	done

eval_winogrande_base_lm3pt3_sft:
	HYDRA_MODE=pos_only accelerate launch --multi_gpu ./lm-eval-no-softmax/lm_eval/__main__.py --batch_size=8 --model=hf-causal-no-softmax --num_fewshot=5 --output_path=results/winogrande_base_lm3pt3-base --model_args=trust_remote_code=true,dtype=bfloat16,pretrained=models/winogrande_base_lm3pt3 --tasks=winogrande ; \
	for num_inst in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 ; do \
		HYDRA_MODE=pos_only accelerate launch --multi_gpu ./lm-eval-no-softmax/lm_eval/__main__.py --batch_size=8 --model=hf-causal-no-softmax --num_fewshot=5 --output_path=results/winogrande_base_lm3pt3-$$num_inst-sft --model_args=trust_remote_code=true,dtype=bfloat16,pretrained=models/winogrande_base_lm3pt3-$$num_inst-s42 --tasks=winogrande ; \
	done

eval_winogrande_base_lm3pt3_negbase:
	HYDRA_MODE=neg_only accelerate launch --multi_gpu ./lm-eval-no-softmax/lm_eval/__main__.py --batch_size=8 --model=hf-causal-no-softmax --num_fewshot=5 --output_path=results/winogrande_base_lm3pt3-base --model_args=trust_remote_code=true,dtype=bfloat16,pretrained=models/winogrande_base_lm3pt3 --tasks=winogrande ; \
	for num_inst in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 ; do \
		HYDRA_MODE=neg_only accelerate launch --multi_gpu ./lm-eval-no-softmax/lm_eval/__main__.py --batch_size=8 --model=hf-causal-no-softmax --num_fewshot=5 --output_path=results/winogrande_base_lm3pt3-$$num_inst-negbase --model_args=trust_remote_code=true,dtype=bfloat16,pretrained=models/winogrande_base_lm3pt3-$$num_inst-s42 --tasks=winogrande ; \
	done
