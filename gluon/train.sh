python train_val.py --rec-train /path/to/train.rec --rec-train-idx /path/to/train.idx \
                    --rec-val /path/to/val.rec --rec-val-idx /path/to/val.idx \
                    --use-rec --batch-size 256 --num-gpus 4 -j 30 --num-epochs 120 --lr-mode cosine --warmup-epochs 5 
