for data in boolq piqa social_i_qa hellaswag winogrande ARC-Easy ARC-Challenge openbookqa 
do
	python commonsense_evaluate.py --model other --base_model 'google/gemma-2b' --adapter LoRA --dataset ${data} --lora_weights ./trained_models/lora_rite_commonsense_15k_rank16_qkvud --batch_size 16 |& tee lora_rite_eval_${data}.log
done
