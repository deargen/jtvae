CKPT_PATH="/aicenter2/mol_generation/ckpts/JT_VAE"

python -u bo/gen_latent.py \
--data $CKPT_PATH/data/zinc/train.txt \
--vocab $CKPT_PATH/data/zinc/vocab.txt \
--model $CKPT_PATH/molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4 \
--sascorer $CKPT_PATH/data/fpscores.pkl.gz \
--hidden 450 --depth 3 --latent 56 \