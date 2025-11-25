### for finetuning

nohup python 2_finetune_mlx.py > mlx_training.log 2>&1


### for checking fine tuned model
 python -m mlx_lm.generate --model Qwen/Qwen2-0.5B --adapter-path ./models/codeact-mlx-qwen2-0.5b --prompt "Your prompt here"