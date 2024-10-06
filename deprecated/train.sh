python wilds_examples/run_expt.py \
    --dataset globalwheat \
    --algorithm ERM \
    --root_dir "/home/guycoh/ReliabilityInML_Project/data/" \
    --train_loader standard \
    --loader_kwargs {'num_workers': 10} \
    --batch_size 6 \
    --model fasterrcnn \
    --n_epochs 100 \
    --optimizer AdamW \
    --device 0 \
    --seed 42 \
    --log_every 10 \
    --use_wandb True 
    # --wandb_api_key_path /home/guycoh/.netrc

