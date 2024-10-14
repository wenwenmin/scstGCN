import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
Image.MAX_IMAGE_PIXELS = None

def main(prefix, login_value):
    def pad(emd, num):
        h, w = emd.shape[0], emd.shape[1]
        pad_h = (num - h % num) % num
        pad_w = (num - w % num) % num

        padded_matrix = np.pad(emd,
                               ((0, pad_h), (0, pad_w), (0, 0)),
                               'constant', constant_values=0)

        new_h, new_w = padded_matrix.shape[:2]
        assert new_h % num == 0 and new_w % num == 0
        return padded_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_emb = []

    def forward_hook(module, input, output):
        features = output[:, 1:, :]
        features = features.cpu().numpy()
        features = features.reshape(features.shape[0], 14, 14, features.shape[2])
        features = np.concatenate(features, axis=1)
        img_emb.append(features)

    login(
        login_value)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    # pretrained=True needed to load UNI weights (and download weights for the first time)
    # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    model.to(device)
    hook = model.norm.register_forward_hook(forward_hook)

    class roi_dataset(Dataset):
        def __init__(self, img,
                     ):
            super().__init__()
            self.transform = transform

            self.images_lst = img

        def __len__(self):
            return len(self.images_lst)

        def __getitem__(self, idx):
            pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
            image = self.transform(pil_image)
            return image

    img = Image.open(f'{prefix}he.jpg')
    img = np.array(img)
    img = pad(img, 224)
    print(f'The size of the histological image is:{img.shape}')
    sub_images = []

    for y in range(0, img.shape[0], 224):
        for x in range(0, img.shape[1], 224):
            sub_image = img[y:y + 224, x:x + 224]
            sub_images.append(sub_image)
    sub_images = np.array(sub_images)

    test_datat = roi_dataset(sub_images)
    database_loader = torch.utils.data.DataLoader(test_datat, batch_size=img.shape[1] // 224, shuffle=False)

    with torch.inference_mode():
        for batch in database_loader:
            feature_emb = model(batch.to(device))
        img_emb = np.concatenate(img_emb, axis=0)
    hook.remove()

    embs = []
    for i in range(img_emb.shape[2]):
        embs.append(img_emb[:, :, i].astype('float32'))

    return embs
