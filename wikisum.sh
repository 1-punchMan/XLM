pretrained_model_path="/home/zchen/XLM/dumped/xlm_en_zh/wie06ycgrm/best-valid_mlm_ppl.pth,/home/zchen/XLM/dumped/xlm_en_zh/wie06ycgrm/best-valid_mlm_ppl.pth"
checkpoint="/home/zchen/XWikiSum/dumped/wikisum/MLM_TLM/TFIDF_paragraph/1_global_fine-tune_enc/checkpoint.pth"
data_path="/home/zchen/XWikiSum/wikisum/en2zh_dataset/MLM_TLM/TFIDF_paragraph/dataset/"
export CUDA_VISIBLE_DEVICES=1
python train.py\
\
    `## main parameters`\
    --exp_name TFIDF_paragraph                                       `# experiment name`\
    --dump_path ./dumped/wikisum/MLM_TLM                                         `# where to store the experiment`\
    --reload_model $pretrained_model_path          `# model to reload for encoder,decoder\
	--reload_checkpoint $checkpoint`\
    \
    `## data location / training objective`\
    --data_path $data_path                           `# data location`\
    --lgs 'en-zh'                                                 `# considered languages`\
    --ws_steps 'en-zh'\
`    # --ae_steps 'en,fr'                                            # denoising auto-encoder training steps
    # --bt_steps 'en-fr-en,fr-en-fr'                                # back-translation steps
    # --word_shuffle 3                                              # noise for auto-encoding loss
    # --word_dropout 0.1                                            # noise for auto-encoding loss
    # --word_blank 0.1                                              # noise for auto-encoding loss
    # --lambda_ae '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient
    `\
    `## transformer parameters`\
    --encoder_only false                                          `# use a decoder for MT`\
    --emb_dim 512                                                `# embeddings / model dimension`\
    --n_layers 6                                                  `# number of layers`\
    --n_heads 8                                                   `# number of heads`\
    --dropout 0.1                                                 `# dropout`\
    --attention_dropout 0.1                                       `# attention dropout`\
    --gelu_activation true                                        `# GELU instead of ReLU
    `\
    `## optimization`\
    `# --tokens_per_batch 2000                                       # use batches with a fixed number of words`\
    --batch_size 4                                               `# batch size (for back-translation)`\
    --bptt 256                                                    `# sequence length`\
    --optimizer adam,lr=0.0001  `# optimizer
    --epoch_size 200000                                           # number of sentences per epoch

    # evaluation`\
    --eval_only false\
    --stopping_criterion 'valid_en-zh_ws_rouge_L,3'                 `# validation metric (when to save the best model)`\
    --validation_metrics 'valid_en-zh_ws_rouge_L'                    `# end experiment if stopping criterion does not improve

    # beam search`\
    --beam_size 4\
    --length_penalty 0.6\
    --given_titles true    `# Given titles when generating sentences.`\
\
    `# XWikisum`\
    --wikisum true\
    --n_paragraphs 10   # # of reference paragraphs to use