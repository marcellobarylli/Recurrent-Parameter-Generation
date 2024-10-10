accelerate launch \
  --main_process_port=29822 \
  --num_processes=1 \
  --gpu_ids='3' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  lstm.py \
