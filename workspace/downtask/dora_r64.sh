accelerate launch \
  --main_process_port=12333 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='3,4' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  dora_r64.py \
