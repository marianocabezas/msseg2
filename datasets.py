import itertools
from copy import deepcopy
import numpy as np
from torch.utils.data.dataset import Dataset
from utils import get_bb


''' Utility function for patch creation '''


def centers_to_slice(voxels, patch_half):
    """
    Function to convert a list of indices defining the center of a patch, to
    a real patch defined using slice objects for each dimension.
    :param voxels: List of indices to the center of the slice.
    :param patch_half: List of integer halves (//) of the patch_size.
    """
    slices = [
        tuple(
            [
                slice(idx - p_len, idx + p_len) for idx, p_len in zip(
                    voxel, patch_half
                )
            ]
        ) for voxel in voxels
    ]
    return slices


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        map(
            lambda t: np.concatenate([np.arange(*t), [t[1]]]),
            zip(min_bb_i, max_bb_i, steps)
        ) for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices


''' Datasets '''


class LongitudinalCroppingDataset(Dataset):
    def __init__(
            self, source, target, activity, masks, patch_size=32,
            overlap=0, filtered=True, balanced=True,
    ):
        # Init
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * 3
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * 3
        else:
            self.overlap = overlap
        self.filtered = filtered
        self.balanced = balanced

        self.source = source
        self.target = target
        self.masks = masks

        self.labels = activity

        # We get the preliminary patch slices (inside the bounding box)...
        slices = get_slices(self.masks, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        if self.filtered:
            if self.balanced:
                self.patch_slices = [
                    (s, i) for i, (label, s_i) in enumerate(
                        zip(activity, slices)
                    )
                    for s in s_i if np.sum(label[s]) > 0
                ]
                self.bck_slices = [
                    (s, i) for i, (label, s_i) in enumerate(
                        zip(activity, slices)
                    )
                    for s in s_i if np.sum(label[s]) == 0
                ]
                self.current_bck = deepcopy(self.bck_slices)
            else:
                self.patch_slices = [
                    (s, i) for i, (label, s_i) in enumerate(
                        zip(activity, slices)
                    )
                    for s in s_i if np.sum(label[s]) > 0
                ]
        else:
            self.patch_slices = [
                (s, i) for i, s_i in enumerate(slices) for s in s_i
            ]

    def __getitem__(self, index):
        if index < (2 * len(self.patch_slices)):
            flip = index >= len(self.patch_slices)
            if flip:
                index -= len(self.patch_slices)
            slice_i, case_idx = self.patch_slices[index]
        else:
            flip = np.random.random() > 0.5
            index = np.random.randint(len(self.current_bck))
            slice_i, case_idx = self.current_bck.pop(index)
            if len(self.current_bck) == 0:
                self.current_bck = deepcopy(self.bck_slices)

        source = self.source[case_idx]
        target = self.target[case_idx]
        labels = self.labels[case_idx]
        none_slice = (slice(None, None),)
        # Patch "extraction".
        data = (
            source[none_slice + slice_i].astype(np.float32),
            target[none_slice + slice_i].astype(np.float32),
        )
        target_data = np.expand_dims(labels[slice_i].astype(np.uint8), axis=0)
        if flip:
            data = (
                np.fliplr(data[0]).copy(),
                np.fliplr(data[1]).copy(),
            )
            target_data = np.fliplr(target_data).copy()

        return data, target_data

    def __len__(self):
        if self.filtered and self.balanced:
            return len(self.patch_slices) * 4
        else:
            return len(self.patch_slices)


class LongitudinalDataset(Dataset):
    def __init__(
        self, source, target, activity, masks
    ):
        # Init
        self.source = source
        self.target = target
        self.masks = masks
        self.labels = activity

    def __getitem__(self, index):
        flip = index >= len(self.labels)
        if flip:
            index -= len(self.labels)

        source = self.source[index]
        target = self.target[index]
        labels = self.labels[index]
        none_slice = (slice(None, None),)
        bb = get_bb(self.masks[index], 2)
        # Patch "extraction".
        data = (
            source[none_slice + bb].astype(np.float32),
            target[none_slice + bb].astype(np.float32),
        )
        target_data = np.expand_dims(labels[bb].astype(np.uint8), axis=0)
        if flip:
            data = (
                np.fliplr(data[0]).copy(),
                np.fliplr(data[1]).copy(),
            )
            target_data = np.fliplr(target_data).copy()

        return data, target_data

    def __len__(self):
        return len(self.labels) * 2


class LongitudinalImageCroppingDataset(Dataset):
    def __init__(
            self, source, target, activity, masks, patch_size=32,
            overlap=0, *args, **kwargs
    ):
        # Init
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * 3
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * 3
        else:
            self.overlap = overlap

        bbs = [get_bb(mask, 2) for mask in masks]
        none_slice = (slice(None, None),)

        self.source = [
            image[none_slice + bb] for image, bb in zip(source, bbs)
        ]
        self.target = [
            image[none_slice + bb] for image, bb in zip(target, bbs)
        ]
        self.masks = [
            mask[none_slice + bb] for mask, bb in zip(masks, bbs)
        ]
        self.labels = [
            mask[none_slice + bb] for mask, bb in zip(activity, bbs)
        ]

        # We get the preliminary patch slices (inside the bounding box)...
        slices = get_slices(self.masks, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        self.patch_slices = [
            (s, i) for i, s_i in enumerate(slices) for s in s_i
        ]

    def __getitem__(self, index):
        flip = index >= len(self.patch_slices)
        if flip:
            index -= len(self.patch_slices)
        slice_i, case_idx = self.patch_slices[index]

        source = self.source[case_idx]
        target = self.target[case_idx]
        none_slice = (slice(None, None),)
        # Patch "extraction".
        data = (
            source[none_slice + slice_i].astype(np.float32),
            target[none_slice + slice_i].astype(np.float32),
        )
        target_data = (
            source[none_slice + slice_i].astype(np.float32),
            target[none_slice + slice_i].astype(np.float32),
        )
        if flip:
            data = (
                np.fliplr(data[0]).copy(),
                np.fliplr(data[1]).copy(),
            )
            target_data = (
                np.fliplr(data[0]).copy(),
                np.fliplr(data[1]).copy(),
            )

        return data, target_data

    def __len__(self):
        return len(self.patch_slices) * 2
