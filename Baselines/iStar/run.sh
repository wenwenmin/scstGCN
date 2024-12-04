#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis
n_genes=1000  # number of most variable genes to impute

# extract histology features
python extract_features.py ${prefix} --device=${device}
# # If you want to retun model, you need to delete the existing results:
# rm ${prefix}embeddings-hist-raw.pickle

# auto detect tissue mask
# If you have a user-defined tissue mask, put it at `${prefix}mask-raw.png` and comment out the line below
python get_mask.py ${prefix}embeddings-hist.pickle ${prefix}mask-small.png

# # segment image by histology features
# python cluster.py --mask=${prefix}mask-small.png --n-clusters=10 ${prefix}embeddings-hist.pickle ${prefix}clusters-hist/
# # # segment image by histology features without tissue mask
# # python cluster.py ${prefix}embeddings-hist.pickle ${prefix}clusters-hist/unmasked/

# select most highly variable genes to predict
# If you have a user-defined list of genes, put it at `${prefix}gene-names.txt` and comment out the line below
python select_genes.py --n-top=${n_genes} "${prefix}cnts.tsv" "${prefix}gene-names.txt"

# train gene expression prediction model and predict at super-resolution
python impute.py ${prefix} --epochs=400 --device=${device}  # train model from scratch
# # If you want to retrain model, you need to delete the existing model:
# rm -r ${prefix}states