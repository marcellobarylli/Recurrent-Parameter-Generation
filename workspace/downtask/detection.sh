accelerate launch \
  --main_process_port=29908 \
  --multi_gpu \
  --num_processes=4 \
  --gpu_ids='0,1,2,5' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  detection.py \
