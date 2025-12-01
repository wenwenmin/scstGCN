import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import rescale
import argparse
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import scanpy as sc
import pickle
from typing import Tuple
from scipy.ndimage import binary_dilation
from matplotlib import pyplot as plt
import cv2
from shapely.geometry import Polygon
from skimage.draw import polygon2mask
import json
from pathlib import Path
import tifffile as tf
import tarfile
Image.MAX_IMAGE_PIXELS = 900000000

def load_10x_matrix_from_tar_gz(tar_file_path):

    tar_path = Path(tar_file_path)

    if not tar_path.exists():
        print(f"Error: File not found {tar_file_path}")
        return None
    
    extract_dir = tar_path.parent / (tar_path.stem.split('.')[0])

    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        
    return extract_dir


def get_xenium_pixel_size(file_path: str):

    xenium_file = Path(file_path)

    if not xenium_file.exists():
        print(f"Error: File not found at {file_path}")
        return None
        
    try:
        with open(xenium_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        pixel_size = data.get("pixel_size")

        if pixel_size is None:
            print(f"Warning: 'pixel_size' key not found in {file_path}")
            return None
            
        return float(pixel_size)

    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not a valid JSON format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")
        return None

def Cal_area_2poly(data1, data2):
    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area
    return inter_area

def unique_pixes(original_list):
    unique_coords = set()

    filtered_list = []

    for inner_list in original_list:
        coord_tuple = tuple(inner_list)

        if coord_tuple not in unique_coords:
            unique_coords.add(coord_tuple)
            filtered_list.append(inner_list)

    return filtered_list

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

def scale_img(raw_image_path, output_image_path, align_dir, scale, shape):
    try:
        he = cv2.imread(raw_image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {raw_image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    new_width = int(he.shape[1] * scale)
    new_height = int(he.shape[0] * scale)

    he_small = cv2.resize(he, (new_width, new_height), interpolation=cv2.INTER_AREA)

    M = np.loadtxt(align_dir, delimiter=',', dtype=float)

    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

    M_new = M.copy()
    M_new[0, 2] *= scale
    M_new[1, 2] *= scale

    dst_width = int(shape[1] * scale)
    dst_height = int(shape[0] * scale)

    img = cv2.warpPerspective(he_small, M_new, dsize=(dst_width, dst_height))

    img = img.astype(np.uint8)
    img = pad_to_multiple_of_224(img)

    mask = np.all(img == [0, 0, 0], axis=-1)
    img[mask] = [235, 235, 235]

    cv2.imwrite(output_image_path, img)
    print(f"Padded image saved to: {output_image_path} (Shape: {img.shape})")
    return img


def create_gene_matrix(folder_path, img_shape: Tuple[int, int, int], gene_matrix_path: str, gene_names_path: str, superpixel_size: int = 16):
    cell_feature_matrix_dir = load_10x_matrix_from_tar_gz(f"{folder_path}cell_feature_matrix.tar.gz")

    adata = sc.read_10x_mtx(f'{cell_feature_matrix_dir}/cell_feature_matrix', var_names='gene_symbols')
    adata.var_names_make_unique()

    pd.Series(adata.var_names).to_csv(gene_names_path, index=False, header=False)
    print(f"Gene names saved to: {gene_names_path}")

    counts = np.array(adata.X.todense())
    boundaries = pd.read_csv(f"{folder_path}cell_boundaries.csv.gz", sep=',', index_col=0)
    boundaries *= 2
    boundaries = boundaries.round().astype(int)
    boundaries = boundaries.reset_index()

    index_map = {}
    current_number = 1
    for value in boundaries.iloc[:, 0]:
        if value not in index_map:
            index_map[value] = current_number
            current_number += 1
    boundaries.iloc[:, 0] = boundaries.iloc[:, 0].map(index_map)
    boundaries = boundaries.set_index('cell_id')

    cells = pd.read_csv(f"{folder_path}cells.csv.gz", sep=',', index_col=0)

    gene_nums = adata.shape[1]
    cnts = np.zeros((img_shape[0] // superpixel_size, img_shape[1] // superpixel_size, gene_nums))

    grid_size = superpixel_size
    for c in range(len(cells)):
        data1 = np.array(boundaries.loc[c + 1])
        list = []
        for i in range(len(data1)):
            list.append(data1[i] // grid_size * grid_size)
        pixes = np.array(unique_pixes(list))

        min_x, min_y = (np.min(data1, axis=0) // grid_size * grid_size).astype(int)
        max_x, max_y = (np.max(data1, axis=0) // grid_size * grid_size).astype(int)

        all_pixes = []
        for x in range(min_x, max_x + grid_size, grid_size):
            for y in range(min_y, max_y + grid_size, grid_size):
                all_pixes.append([x, y])

        all_pixes = np.array(all_pixes)
        mask = polygon2mask((max_y - min_y + grid_size, max_x - min_x + grid_size), (data1 - [min_x, min_y]) / grid_size)
        valid_pixes = all_pixes[mask[all_pixes[:, 1] // grid_size - min_y // grid_size, all_pixes[:, 0] // grid_size - min_x // grid_size]]

        pixes = np.unique(np.vstack((pixes, valid_pixes)), axis=0)

        areas = []
        for p in pixes:
            x, y = p[0], p[1]
            data2 = np.array([[x, y], [x, y + 16], [x + 16, y + 16], [x + 16, y]])
            area2 = Cal_area_2poly(data1, data2)
            areas.append(area2)

        area1 = sum(areas)
        for i, p in enumerate(pixes):
            x, y = p[0], p[1]
            ratio = areas[i] / area1
            gene = counts[c, :] * ratio
            w, h = int(x // 16), int(y // 16)
            if h >= cnts.shape[0] or w >= cnts.shape[1]:
                continue
            cnts[h, w, :] += gene

    with open(gene_matrix_path, 'wb') as f:
        pickle.dump(cnts, f)
        print(f"3D Gene Matrix (shape {cnts.shape}) saved to: {gene_matrix_path}")
    return cnts

def get_mask(he, dir):
    h, w, _ = he.shape

    block_size = 16

    h_new, w_new = h // block_size, w // block_size

    mask = np.zeros((h_new, w_new), dtype=bool)

    for i in range(h_new):
        for j in range(w_new):
            block = he[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, 1]
            mask[i, j] = np.mean(block) <= 230

    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    plt.imsave(f'{dir}mask.png', mask, cmap='gray')
    return mask

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
        gene_names = [line.strip() for line in file] # series

    cnts_df = pd.DataFrame(data=cnts, columns=gene_names)
    cnts_df.to_csv(f'{dir}cnts.csv')

def process_Xenium(
    data_dir: str, 
    sample_subfolder: str, 
    align_name: str, 
    raw_image_name: str, 
    output_image_name: str, 
    gene_names_name: str,
    ground_truth_name: str,
    scale_pix_size: float = 0.5,
    superpixel_size: int = 16,
    pseudo_radius: int = 55
):   
    os.makedirs(data_dir, exist_ok=True)
    
    folder_path = os.path.join(data_dir, sample_subfolder) # ../..outs/
    
    raw_image_path = os.path.join(data_dir, raw_image_name)
    output_image_path = os.path.join(data_dir, output_image_name)
    gene_names_path = os.path.join(data_dir, gene_names_name)
    gene_matrix_path = os.path.join(data_dir, ground_truth_name)
    raw_pix_size_path =  os.path.join(folder_path, 'experiment.xenium')
    align_dir = os.path.join(data_dir, align_name)

    print(f"--- Start preprocessing Xenium data ---")

    raw_pix_size = get_xenium_pixel_size(raw_pix_size_path)

    with open(f'{data_dir}/pixel-size-raw.txt', 'w') as f:
        f.write(str(raw_pix_size))
    with open(f'{data_dir}/pixel-size.txt', 'w') as f:
        f.write(str(scale_pix_size))

    scale = raw_pix_size / scale_pix_size

    print(f"Original pixel size: {raw_pix_size:.4f} μm. Scaling factor: {scale:.4f}")

    with tf.TiffFile(f'{folder_path}morphology.ome.tif') as tif:
        morphology_ome_shape = tif.series[0].shape[1:]

    img_padded = scale_img(raw_image_path, output_image_path, align_dir, scale, shape=morphology_ome_shape)

    genes_3D = create_gene_matrix(
        folder_path=folder_path,
        img_shape=img_padded.shape,
        gene_matrix_path=gene_matrix_path,
        gene_names_path=gene_names_path,
        superpixel_size=superpixel_size
    )

    mask = get_mask(img_padded, data_dir)

    get_pseudo_visium_locs(img_padded, mask, data_dir)

    get_pseudo_visium_cnts(genes_3D, pseudo_radius / superpixel_size, data_dir, superpixel_size)

    with open(f'{data_dir}/radius.txt', 'w') as f:
        f.write(str(pseudo_radius))
    
    print("\n--- All processing steps completed successfully ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="End-to-end Xenium processing."
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='/home/nas3/biod/xueshuailin/data/super-resolution-task/Xenium/demo/', 
        help='Base directory containing all data (e.g., "/../mouse-brain/").'
    )
    parser.add_argument(
        '--sample_size', 
        type=str, 
        default='Xenium_V1_hLiver_nondiseased_section_FFPE_outs/'
    )
    parser.add_argument(
        '--align_name', 
        type=str, 
        default='Xenium_V1_hLiver_nondiseased_section_FFPE_he_imagealignment.csv',
        help='Name of the scale matrix image file within data_dir.'
    )
    parser.add_argument(
        '--raw_image_name', 
        type=str, 
        default='Xenium_V1_hLiver_nondiseased_section_FFPE_he_image.ome.tif', 
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
        default='gene-names.txt'
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
        '--pseudo_radius', 
        type=int, 
        default=55, 
        help='The radius of pseudo visium data in pixels.'
    )

    args = parser.parse_args()

    process_Xenium(
        data_dir=args.data_dir,
        sample_subfolder=args.sample_size,
        align_name=args.align_name,
        raw_image_name=args.raw_image_name,
        output_image_name=args.output_image_name,
        gene_names_name=args.gene_names_name,
        ground_truth_name=args.ground_truth,
        scale_pix_size=args.scale_pix_size,
        superpixel_size=args.superpixel_size,
        pseudo_radius=args.pseudo_radius
    )