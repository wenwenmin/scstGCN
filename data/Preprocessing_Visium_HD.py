import bin2cell as b2c
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import rescale
import argparse
import os
import scanpy as sc
import pickle
from typing import Tuple
from scipy.ndimage import binary_dilation
from matplotlib import pyplot as plt
Image.MAX_IMAGE_PIXELS = 900000000

def pad_to_multiple_of_224(img: np.ndarray, fill_value: int = 255) -> np.ndarray:
    """Pads an image (H, W, C) to dimensions that are a multiple of 224."""
    H, W = img.shape[:2]
    new_H = ((H + 223) // 224) * 224
    new_W = ((W + 223) // 224) * 224

    pad_H = new_H - H
    pad_W = new_W - W

    img_padded = np.pad(
        img,
        ((0, pad_H), (0, pad_W), (0, 0)),
        mode='constant',
        constant_values=fill_value
    )
    return img_padded

def get_disk_mask(radius, boundary_width=None):
    radius_ceil = np.array(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin

def get_locs(dir, rescale_factor):
    locs = pd.read_csv(f'{dir}locs.csv', header=0, index_col=0)
    locs = np.stack([locs['x'], locs['y']], -1)
    locs = locs.astype(float)

    locs /= rescale_factor

    locs = locs.round().astype(int)

    return locs

def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)

    center = shape // 2
    r = np.stack([-center, shape-center], -1)
    x_list = []
    for s in locs:
        patch = img[
                s[1]+r[0][0]:s[1]+r[0][1],
                s[0]+r[1][0]:s[0]+r[1][1]]
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list

def create_gene_matrix(adata: sc.AnnData, img_shape: Tuple[int, int, int], gene_matrix_path: str, gene_names_path: str, superpixel_size: int = 16, num_top_genes: int = 1000):
    H, W, _ = img_shape
    
    a = int(H // superpixel_size)
    b = int(W // superpixel_size)
    
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=num_top_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    
    print(f"Selected {adata.n_vars} highly variable genes.")

    pd.Series(adata.var_names).to_csv(gene_names_path, index=False, header=False)
    print(f"Gene names saved to: {gene_names_path}")

    genes_3D = np.zeros((a, b, adata.n_vars), dtype=np.float32)
    X = adata.X.toarray()
    num_spots = adata.n_obs
    
    pxl_row = adata.obs['pxl_row_in_fullres']
    pxl_col = adata.obs['pxl_col_in_fullres']
    
    for k in range(num_spots):
        i = (pxl_row.iloc[k] - (superpixel_size // 2)) // superpixel_size
        j = (pxl_col.iloc[k] - (superpixel_size // 2)) // superpixel_size

        if 0 <= i < a and 0 <= j < b:
            genes_3D[i, j, :] = -1 if np.all(X[k, :] == 0) else X[k, :]
        
    with open(gene_matrix_path, 'wb') as f:
        pickle.dump(genes_3D, f)
    print(f"3D Gene Matrix (shape {genes_3D.shape}) saved to: {gene_matrix_path}")
    return genes_3D

def get_mask(genes, dir):
    mask = np.any(genes, axis=2)
    mask = mask.astype(np.uint8) * 255

    bool_mask = mask == 255
    dilated_mask = binary_dilation(bool_mask)
    new_mask = dilated_mask.astype(np.uint8) * 255

    plt.imsave(f'{dir}mask.png', new_mask, cmap='gray')
    return new_mask

def get_pseudo_visium_locs(
    he: Image, 
    mask_raw: Image, 
    dir: str, 
    distance: int = 100, 
    pixel_size: float = 0.5
):
    W_px, H_px = he.shape[1], he.shape[0]
    W_um, H_um = W_px * pixel_size, H_px * pixel_size
    
    locations_um = []

    for i in range(0, int(W_um - 27.5), distance):
        k = 0 if (i / distance) % 2 == 0 else 50
        
        for j in range(0, int(H_um - 27.5 - (127.5 - k)), distance):
            x = i + 27.5
            y = j + 27.5 + k
            locations_um.append([x, y])

    if not locations_um:
        print("Warning: No locations were generated. Check distance and image size.")
        return

    locations_um_np = np.array(locations_um, dtype=np.float32)

    locations_px = (locations_um_np / pixel_size).astype(np.int32)
    df = pd.DataFrame(locations_px, columns=['x', 'y'])
    print(f"Generated {len(df)} initial locations.")

    mask = (mask_raw > 0)
    H_mask, W_mask = mask.shape
    
    rows_to_delete = []

    df['mask_col'] = df['x'] // 16 - 3 # j
    df['mask_row'] = df['y'] // 16 - 3 # i

    df_filtered = df[
        (df['mask_row'] >= 0) & (df['mask_row'] < H_mask) &
        (df['mask_col'] >= 0) & (df['mask_col'] < W_mask)
    ].copy()

    mask_indices = df_filtered[['mask_row', 'mask_col']].values
    
    mask_values = mask[mask_indices[:, 0], mask_indices[:, 1]]
    
    df_final = df_filtered[mask_values].copy()

    df_final['spots'] = range(len(df_final))
    df_final[['spots', 'x', 'y']].to_csv(f'{dir}locs.csv', index=False, float_format='%.2f')

def get_pseudo_visium_cnts(genes, radius, dir, superpixel_size):
    genes[genes < 0] = 0

    mask = get_disk_mask(radius)
    locs =get_locs(dir, superpixel_size)

    x = get_patches_flat(genes, locs, mask)

    cnts = np.sum(x, axis=1)

    with open(f"{dir}gene-names.txt", 'r') as file:
        gene_names = [line.strip() for line in file]

    cnts_df = pd.DataFrame(data=cnts, columns=gene_names)
    cnts_df.to_csv(f'{dir}cnts.csv')


def process_visium_HD(
    data_dir: str, 
    sample_subfolder: str, 
    raw_image_name: str, 
    output_image_name: str, 
    gene_names_name: str,
    ground_truth_name: str,
    scale_pix_size: float = 0.5,
    superpixel_size: int = 16,
    num_top_genes: int = 1000,
    pseudo_radius: int = 55
):   
    os.makedirs(data_dir, exist_ok=True)
    
    folder_path = os.path.join(data_dir, sample_subfolder)
    
    raw_image_path = os.path.join(data_dir, raw_image_name)
    output_image_path = os.path.join(data_dir, output_image_name)
    gene_names_path = os.path.join(data_dir, gene_names_name)
    gene_matrix_path = os.path.join(data_dir, ground_truth_name)

    print(f"--- Start preprocessing Visium HD data ---")

    try:
        adata = b2c.read_visium(folder_path)
    except Exception as e:
        print(f"Error loading Visium data: {e}")
        return

    adata = adata[(adata.obs >= 0).all(axis=1)].copy()

    visium_key = next(iter(adata.uns['spatial'])) 
    raw_pix_size = adata.uns['spatial'][visium_key]['scalefactors']['microns_per_pixel']

    with open(f'{data_dir}/pixel-size-raw.txt', 'w') as f:
        f.write(str(raw_pix_size))
    with open(f'{data_dir}/pixel-size.txt', 'w') as f:
        f.write(str(scale_pix_size))

    scale = raw_pix_size / scale_pix_size

    print(f"Original pixel size: {raw_pix_size:.4f} μm. Scaling factor: {scale:.4f}")

    adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']] = adata.obsm["spatial"] * scale
    adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']] = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].round().astype(int)

    try:
        img = np.array(Image.open(raw_image_path))
    except FileNotFoundError:
        print(f"Error: Image file not found at {raw_image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    img = rescale(img, [scale, scale, 1], preserve_range=True)
    img = img.astype(np.uint8)
    img_padded = pad_to_multiple_of_224(img)
    Image.fromarray(img_padded).save(output_image_path)
    print(f"Padded image saved to: {output_image_path} (Shape: {img_padded.shape})")

    genes_3D = create_gene_matrix(
        adata=adata,
        img_shape=img_padded.shape,
        gene_matrix_path=gene_matrix_path,
        gene_names_path=gene_names_path,
        superpixel_size=superpixel_size,
        num_top_genes=num_top_genes
    )

    mask = get_mask(genes_3D, data_dir)

    get_pseudo_visium_locs(img_padded, mask, data_dir)

    get_pseudo_visium_cnts(genes_3D, pseudo_radius / superpixel_size, data_dir)

    with open(f'{data_dir}/radius.txt', 'w') as f:
        f.write(str(pseudo_radius))
    
    print("\n--- All processing steps completed successfully ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="End-to-end Visium HD processing."
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='/home/nas3/biod/xueshuailin/data/super-resolution-task/HD/demo/', 
        help='Base directory containing all data (e.g., "/../mouse-brain/").'
    )
    parser.add_argument(
        '--sample_size', 
        type=str, 
        default='square_008um',
        help='Subfolder within data_dir containing Visium HD output (e.g., "square_008um").'
    )
    parser.add_argument(
        '--raw_image_name', 
        type=str, 
        default='Visium_HD_Mouse_Brain_tissue_image.tif', 
        help='Name of the image file within data_dir.'
    )
    
    parser.add_argument(
        '--output_image_name', 
        type=str, 
        default='he.jpg', 
        help='Name for the final scaled image.'
    )
    parser.add_argument(
        '--gene_names_name', 
        type=str, 
        default='gene-names.txt', 
        help='Name for the output highly variable gene names file (default: "gene-names.txt").'
    )
    parser.add_argument(
        '--ground_truth', 
        type=str, 
        default='genes_3D.pkl', 
        help='Name for the ground truth (default: "genes_3D.pkl").'
    )
    
    parser.add_argument(
        '--scale_pix_size', 
        type=float, 
        default=0.5, 
        help='The desired pixel size in microns (default: 0.5).'
    )
    parser.add_argument(
        '--superpixel_size', 
        type=int, 
        default=16, 
        help='The side length of the superpixel in pixels (default: 16).'
    )
    parser.add_argument(
        '--num_top_genes', 
        type=int, 
        default=1000, 
        help='Number of highly variable genes to select (default: 1000).'
    )

    parser.add_argument(
        '--pseudo_radius', 
        type=int, 
        default=55, 
        help='The radius of pseudo visium data in pixels.'
    )

    args = parser.parse_args()

    process_visium_HD(
        data_dir=args.data_dir,
        sample_subfolder=args.sample_size,
        raw_image_name=args.raw_image_name,
        output_image_name=args.output_image_name,
        gene_names_name=args.gene_names_name,
        ground_truth_name=args.ground_truth,
        scale_pix_size=args.scale_pix_size,
        superpixel_size=args.superpixel_size,
        num_top_genes=args.num_top_genes,
        pseudo_radius=args.pseudo_radius
    )