CKPT_PATH="/aicenter2/mol_generation/ckpts/JT_VAE"
SEED=1

python -u bo/run_bo.py \
--vocab $CKPT_PATH/data/zinc/vocab.txt \
--model $CKPT_PATH/molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4 \
--features $CKPT_PATH/data/latent_label_dict.pt \
--save_dir results$SEED \
--hidden 450 --depth 3 --latent 56 --seed $SEED