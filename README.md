# scstGCN
## Overview
Ideal ST data should have single-cell resolution and cover the entire tissue surface,but generating such ST data with existing platforms remains challenging scstGCN is a GCN-based method that leverages a weakly supervised learning framework to integrate multimodal information and then infer super-resolution gene expression at single-cell level. It first extract high-resolution multimodal feature map, including histological feature map, positional feature map, and RGB feature map. and then use the GCN module to predict super-resolution gene expression from multimodal feature map by a weakly supervised framework. scstGCN can predict super-resolution gene expression accurately, aid researchers in discovering biologically meaningful differentially expressed genes and pathways. Additionally, it can predict expression both outside the spots and in external tissue sections.

![Overview.png](Overview.png)

## Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 3090)
- `pip install -r requiremnt.txt`

## Data
All the datasets used in this paper can be downloaded from url：[https://zenodo.org/records/12800375](https://zenodo.org/records/12800375).
### Data format
- `he-raw.jpg`: The original histological image.
- `cnts.csv`: Spot-based gene expression data, where each row represents a spot and each column represents a gene.
- `locs-raw.csv`: All spots’ two-dimensional coordinates information, where each row represents a spot corresponding to `cnts.csv`. The first and second columns in this files represent x-coordinate and y-coordinate, respectively. The units of coordinates information are pixels corresponding to the histological image.
- `pixel-size-raw.txt`: The actual physical size corresponding to each pixel in the histological image, measured in micrometers.
- `radius-raw.txt`: The number of pixels in histological image corresponding to the radius of a spot.
### Data preprocessing
If you want to experiment with Visium HD data at single-cell resolution, you need to go through the following steps to get the spot-based Pseudo-ST data:
- [get_pseudo_loc.py](get_pseudo_loc.py): You can obtain the position coordinates of the spot-based Pseudo-ST data through this script. The entire detected tissue will be covered by Pseudo-spots. Depending on the data characteristics, you may need to adjust the diameter variable in this script.
- [pixel_to_spot.py](pixel_to_spot.py): You can obtain the spatial gene expression of the spot-based Pseudo-ST data through this script. The genes of all superpixels covered by the spot will be summed as its gene expression. 

## Getting access
In our multimodal feature mapping extractor, the ViT architecture utilizes a self-pretrained model called UNI. You need to request access to the model weights from the Huggingface model page at:[https://huggingface.co/mahmoodlab/UNI](https://huggingface.co/mahmoodlab/UNI). It is worth noting that you need to apply for access to UNI login and replace it in the [demo.ipynb](demo.ipynb).

## Running demo
We provide a examples for predicting super-resolution gene expression data of 10X Visium human dorsolateral prefrontal cortex tissue, please refer to [demo.ipynb](demo.ipynb).

## Baselines
We have listed the sources of some representative baselines below, and we would like to express our gratitude to the authors of these baselines for their generous sharing.

[iStar](https://github.com/daviddaiweizhang/istar).

- iStar models super-resolution gene expression from hierarchical histological features using a feedforward neural network. This method is divided into two parts: the HIPT model \cite{hipt} is used to extract hierarchical histological features, and then a feedforward neural network is used to predict super-resolution gene expression. iStar emphasizes multi-layer histological feature extraction. Each superpixel contains not only local cell feature information, but also global relationship information in the entire histology image.

- XFuse integrates Spatial transcriptomics (ST) data and histology images using a deep generative model to infer super-resolution gene expression profiles. This method considers spatial gene expression and histological image data as observable effects of potential tissue states, and maps image data to potential states through a recognition neural network. XFuse performs well in the top-ranked highly expressed genes, but is misled by intense morphological similarities between different regions in histology image, resulting in poor prediction of low-expression regions of genes.
  
- TESLA generates high-resolution gene expression profiles based on Euclidean distance metric, which considers the similarity in physical locations and histology image features between superpixels and measured spots. TESLA generates super-resolution gene expression based on the assumption that the expression patterns for spatially variable genes are correlated with histology image features. Therefore, it is likely to perform poorly on non-spatially variable genes.
- STAGE to generate gene expression data for unmeasured spots or points from Spatial Transcriptomics with a spatial location-supervised Auto-encoder GEnerator by integrating spatial information and gene expression data. STAGE was originally designed to predict gene expression at spot gaps, but we were also able to obtain super-resolution gene expression using STAGE by converting spatial coordinates from the spot level to the superpixel level.

    

## Acknowledgements
Part of the code, such as the training framework based on pytorch lightning and the method for mask image in this repository is adapted from the [iStar](https://github.com/daviddaiweizhang/istar). And the Vision Transformer in this repository has been pre-trained by [UNI](https://github.com/mahmoodlab/UNI). We are grateful to the authors for their excellent work.

## Contact details
If you have any questions, please contact xueshuailin@163.com.


### Citing
<p>The corresponding BiBTeX citation are given below:</p>
<div class="highlight-none"><div class="highlight"><pre>
@article{xue2024inferring,
  title={Inferring single-cell resolution spatial gene expression via fusing spot-based spatial transcriptomics, location and histology using GCN},
  author={Xue, Shuailin and Zhu, Fangfang and Chen, Jinyu and Min, Wenwen},
  journal={Briefings in Bioinformatics},
  volume={DOI:10.1093/bib/bbae630},
  year={2024},
  publisher={Oxford University Press}
}
</pre></div>
