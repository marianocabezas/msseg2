import time
import argparse
import os
from time import strftime
import numpy as np
import nibabel as nib
from nibabel import load as load_nii
from skimage.measure import label as bwlabeln
from scipy.ndimage.morphology import binary_dilation
from utils import color_codes
from utils import get_mask, get_normalised_image
from utils import time_to_string, find_file
from models import NewLesionsUNet
from utils import remove_small_regions

"""
> Arguments
"""


def parse_inputs():
    """
    Arguments for the different lesion activity analysis pipelines.
    """
    parser = argparse.ArgumentParser(
        description='Run the longitudinal MS lesion segmentation pipelines.'
    )
    parser.add_argument(
        '-t', '--test-directory',
        dest='test_dir', default='/data',
        help='Option to use leave-one-out. The second parameter is the '
             'folder with all the patients.'
    )
    parser.add_argument(
        '-m', '--model-directory',
        dest='model_dir', default='./ModelWeights',
        help='Path to the model.'
    )
    parser.add_argument(
        '-b', '--batch-size',
        dest='batch_size',
        type=int, default=512,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-w', '--patch-size',
        dest='patch_size',
        type=int, default=32,
        help='Patch size'
    )
    parser.add_argument(
        '-d', '--baseline-dilation',
        dest='base_dilation',
        type=int, default=1,
        help='Patch size'
    )
    return vars(parser.parse_args())


"""
> Network functions
"""


def test(n_folds=5, verbose=0):
    # Init
    c = color_codes()
    options = parse_inputs()
    t_path = options['test_dir']
    model_path = options['model_dir']

    net = NewLesionsUNet(n_images=1)
    n_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print('Unet parameters {:d}'.format(n_params))

    brain_name = 'brain_mask.nii.gz'
    bl_name = 'bl_flair.nii.gz'
    bl_lesion_name = 'bl_lesion.nii.gz'
    fu_name = 'fu_flair.nii.gz'
    fu_lesion_name = 'fu_lesion.nii.gz'
    activity_name = 'positive_activity.nii.gz'
    enlarging_name = 'enlarging_activity.nii.gz'
    new_name = 'new_activity.nii.gz'

    case_start = time.time()

    pbrain_name = find_file(brain_name, t_path)
    brain = get_mask(pbrain_name)
    pbl_name = find_file(bl_name, t_path)
    pfu_name = find_file(fu_name, t_path)
    bl = np.expand_dims(
        get_normalised_image(pbl_name, brain, dtype=np.float16), axis=0
    )
    fu = np.expand_dims(
        get_normalised_image(pfu_name, brain, dtype=np.float16), axis=0
    )

    ref_nii = load_nii(pbrain_name)
    segmentation = np.zeros_like(ref_nii.get_fdata())

    for fi in range(n_folds):
        net_name = 'positive-unet_n{:d}.pt'.format(fi)
        net.load_model(os.path.join(model_path, net_name))
        if verbose > 1:
            print(
                '{:}Runing activity pipeline {:}(fold: {:})'.format(
                    c['clr'] + c['c'], c['nc'],
                    c['c'] + str(fi) + c['nc']
                ), end='\r'
            )

        batch_size = options['batch_size']
        patch_size = options['patch_size']
        brain_bin = brain.astype(np.bool)
        idx = np.where(brain_bin)
        bb = tuple(
            slice(min_i, max_i)
            for min_i, max_i in zip(
                np.min(idx, axis=-1), np.max(idx, axis=-1)
            )
        )
        seg = np.zeros(brain.shape)
        t_source = bl[(slice(None),) + bb]
        t_target = fu[(slice(None),) + bb]
        seg_bb = net.new_lesions_patch(
            t_source, t_target, patch_size, batch_size
        )
        seg[bb] = seg_bb

        seg_temp = np.zeros_like(ref_nii.get_fdata())
        seg_temp[bb] = seg_bb
        seg_temp[np.logical_not(brain_bin)] = 0

        segmentation += (seg_temp / n_folds)

    pactivity_name = os.path.join(t_path, activity_name)
    penlarging_name = os.path.join(t_path, enlarging_name)
    pnew_name = os.path.join(t_path, new_name)
    # Thresholding + brain mask filtering
    segmentation_bin = remove_small_regions(
        np.logical_and(segmentation > 0.5, brain.astype(np.bool)),
        min_size=2
    )

    segmentation_lab = bwlabeln(segmentation_bin)

    # Baseline filtering
    pbl_lesion_name = os.path.join(t_path, bl_lesion_name)
    bl_lesion = get_mask(pbl_lesion_name, dtype=np.bool)

    lesions = [
        (lab, segmentation_lab == lab)
        for lab in np.unique(segmentation_lab)
    ]
    bl_overlap = [
        (lab, np.sum(np.logical_and(lesion, bl_lesion)) / np.sum(lesion))
        for lab, lesion in lesions
    ]
    bl_valid = [
        lab
        for lab, overlap in bl_overlap
        if lab > 0 and overlap < 0.5
    ]

    segmentation_bl = np.isin(segmentation_lab, bl_valid)

    # Follow-up filtering
    pfu_lesion_name = os.path.join(t_path, fu_lesion_name)
    fu_lesion = get_mask(pfu_lesion_name, dtype=np.bool)
    fu_valid = segmentation_lab[fu_lesion]
    segmentation_fu = np.isin(segmentation_lab, fu_valid[fu_valid > 0])

    # Final mask
    final_activity = np.logical_and(segmentation_fu, segmentation_bl)
    segmentation_nii = nib.Nifti1Image(
        final_activity, ref_nii.get_qform(), ref_nii.header
    )
    segmentation_nii.to_filename(pactivity_name)
    
    # New / Enlarging split
    dilated_bl_mask = binary_dilation(
        bl_lesion, iterations=options['base_dilation']
    )
    activity_label, n_activity = bwlabeln(
        final_activity, return_num=True
    )
    all_labels = np.unique(activity_label)
    all_labels = all_labels[all_labels > 0]
    enlarging_labels = np.unique(activity_label[dilated_bl_mask])
    enlarging_labels = enlarging_labels[enlarging_labels > 0]
    new_labels = [
        label for label in all_labels if label not in enlarging_labels
    ]
    n_enlarging = len(enlarging_labels)
    n_new = n_activity - n_enlarging
    # Enlarging
    enlarging_activity = np.isin(activity_label, enlarging_labels)
    segmentation_nii = nib.Nifti1Image(
        enlarging_activity, ref_nii.get_qform(), ref_nii.header
    )
    segmentation_nii.to_filename(penlarging_name)
    # New
    new_activity = np.isin(activity_label, new_labels)
    segmentation_nii = nib.Nifti1Image(
        new_activity, ref_nii.get_qform(), ref_nii.header
    )
    segmentation_nii.to_filename(pnew_name)

    with open(os.path.join(t_path, 'activity_number.txt'), 'w') as f:
        f.write(
            'new,{:d},{:d}\nenlarging,{:d},{:d}'.format(
                n_new, np.sum(new_activity),
                n_enlarging, np.sum(enlarging_activity)
            )
        )

    if verbose > 1:
        time_elapsed = time.time() - case_start
        print(
            '{:}Finished activity pipeline {:}'.format(
                c['clr'] + c['c'],
                c['nc'] + time_to_string(time_elapsed)
            )
        )


"""
> Dummy main function
"""


def main(verbose=2):
    # Init
    c = color_codes()
    print(
        '{:}[{:}] {:}<Activity testing pipeline>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    ''' <Segmentation task> '''
    test(verbose=verbose)


if __name__ == '__main__':
    main()
