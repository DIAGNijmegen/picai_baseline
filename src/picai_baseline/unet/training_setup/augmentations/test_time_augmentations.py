import torch
from batchgenerators.augmentations.spatial_transformations import augment_zoom, augment_resize


# Horizontal Flip for 5D Tensors (B,C,W,H,D)
def hflip_tensor(tensor):
    return tensor.flip(-3)

# Simulate Low-Res Scan for 5D Tensors (B,C,H,W,D)


def sim_low_res_tensor(tensor):
    img_npy = tensor.detach().numpy()[0]
    og_shape = img_npy.shape

    # Downsample Image by 12.5%
    img_npy, _ = augment_zoom(img_npy, None, (0.875, 0.875, 1.000))

    # Upsample to Original Size via Nearest Neighbor Interpolation
    img_npy, _ = augment_resize(img_npy, None, og_shape[0], order=0)

    return torch.unsqueeze(torch.from_numpy(img_npy.copy()), 0)
