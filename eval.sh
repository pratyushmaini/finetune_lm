cd ~/scratch/projects/llm-foundry/scripts/inference
python convert_composer_to_hf.py --composer_path ../train/piqa/checkpoints/ep1-ba167-rank0.pt --hf_output_path ../train/piqa_hf/ --output_precision fp32

cd /home/pratyus2/scratch/projects/finetune_lm/
CUDA_VISIBLE_DEVICES=6 python eval_mcq.py piqa