python inference.py --output_dir ./outputs/hfbert/ \
                    --dataset_name ../data/test_dataset/ \
                    --model_name_or_path ./models/train_dataset/ \
                    --do_predict --use_dense \
                    --top_k_retrieval 50 \
                    --encoder_base "klue/bert-base" \
                    --p_encoder_ckpt "./encoders/hfbert/p_encoder_hfbert_bs_16_e_5" \
                    --q_encoder_ckpt "./encoders/hfbert/q_encoder_hfbert_bs_16_e_5" \
                    --use_HFBert \
                    --retriever "dpr"