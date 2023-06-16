python inference.py --output_dir ./outputs/hfbert3-50/ \
                    --dataset_name ../data/test_dataset/ \
                    --model_name_or_path ./models/train_dataset/ \
                    --do_predict --use_dense \
                    --top_k_retrieval 50 \
                    --encoder_base "klue/bert-base" \
                    --p_encoder_ckpt "./encoders/val_best/p_encoder" \
                    --q_encoder_ckpt "./encoders/val_best/q_encoder" \
                    --retriever "dpr" \
                    --use_HFBert True