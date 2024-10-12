accelerate launch \
  --main_process_port=12314 \
  --num_processes=4 \
  --gpu_ids='1,2,3,4' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  vitsmall_16384.py \
