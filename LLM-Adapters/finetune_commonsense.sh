optim=lora_rite
data=commonsense_15k
data_path=ft-training_set/commonsense_15k.json
rank=16
for lr in 2e-4 5e-4 1e-3
do
	python finetune.py --base_model 'google/gemma-2b' --data_path ${data_path} --output_dir ./trained_models/${optim}_${lr}_${data}_rank${rank}_qkvud --batch_size 64 --micro_batch_size 4 --learning_rate ${lr} --cutoff_len 512 --val_set_size 120 --eval_step 80 --save_step 80 --lora_r ${rank} --target_modules="q_proj, k_proj, v_proj, up_proj, down_proj" --train_on_inputs True --optim ${optim}  |& tee ${optim}_${lr}_${data}_rank${rank}_qkvud.log
done
