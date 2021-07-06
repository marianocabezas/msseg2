import time
import argparse
import os
from time import strftime
import SimpleITK as itk
import numpy as np
import nibabel as nib
from skimage.measure import label
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_closing
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from utils import color_codes, get_normalised_image, find_file
from utils import remove_small_regions, remove_boundary_regions
from models import NewLesionsAttUNet


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
        '-m', '--model-directory',
        dest='model_dir', default='/model',
        help='Path to the model.'
    )
    parser.add_argument(
        '-t', '--test-directory',
        dest='test_dir', default=os.getcwd(),
        help='Option to use leave-one-out. The second parameter is the '
             'folder with all the patients.'
    )
    parser.add_argument(
        '-t1', '--time1',
        dest='t1_name',
        default='flair_time01_on_middle_space.nii.gz',
        help='Baseline image name'
    )
    parser.add_argument(
        '-t2', '--time2',
        dest='t2_name',
        default='flair_time02_on_middle_space.nii.gz',
        help='Follow-up image name'
    )
    parser.add_argument(
        '-o', '--output',
        dest='out_name',
        default='output_segmentation.nii.gz',
        help='Output segmentation name'
    )
    return vars(parser.parse_args())


"""
> Pre-processing functions
"""


def itkn4(input_name, output_name, max_iters=400, levels=3):
    """
    :param input_name:
    :param output_name:
    :param max_iters:
    :param levels:
    :return:
    """

    # Init
    image = itk.ReadImage(input_name)
    mask = itk.OtsuThreshold(image, 0, 1, 200)

    image = itk.Cast(image, itk.sitkFloat32)
    corrector = itk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([max_iters] * levels)
    output = corrector.Execute(image, mask)
    itk.WriteImage(output, output_name)


"""
> Network functions
"""


def test(n_folds=5, verbose=0):
    # Init
    c = color_codes()
    options = parse_inputs()
    t_path = options['test_dir']
    model_path = options['model_dir']
    tmp_path = '/tmp'
    batch_size = 512
    patch_size = 32

    brain_name = 'brain_mask.nii.gz'
    bl_name = options['t1_name']
    fu_name = options['t2_name']
    activity_name = options['out_name']

    global_start = time.time()
    # Preprocessing
    time_str = time.strftime("%H:%M:%S")
    print(
        '{:}[{:}]{:} Preprocessing the image{:}'.format(
            c['c'], time_str, c['g'], c['nc'])
    )

    bl_raw = os.path.join(t_path, bl_name)
    bl_final = os.path.join(tmp_path, 'time1_corrected.nii.gz')
    print('- Correcting FLAIR [time 1]', end='\r')
    itkn4(bl_raw, bl_final)

    fu_raw = os.path.join(t_path, fu_name)
    fu_final = os.path.join(tmp_path, 'time2_corrected.nii.gz')
    print('- Correcting FLAIR [time 2]', end='\r')
    itkn4(fu_raw, fu_final)

    print('- Brain mask', end='\r')
    fu = nib.load(fu_final).get_fdata()
    bl = nib.load(bl_final).get_fdata()
    fu_mask = fu > threshold_otsu(fu)
    bl_mask = bl > threshold_otsu(bl)

    mask = binary_erosion(np.logical_and(bl_mask, fu_mask), iterations=5)
    mask_cc = label(mask)
    mask_lab = np.argmax(np.bincount(mask_cc.flat)[1:]) + 1
    mask = binary_dilation(mask_cc == mask_lab, iterations=5)
    brain = binary_fill_holes(binary_closing(mask, iterations=20))
    final_brain = np.zeros_like(brain)
    slice_vol = brain.shape[0] * brain.shape[1]
    for s in range(brain.shape[-1]):
        brain_slice = brain[:, :, s]
        brain_ratio = np.sum(brain_slice) / slice_vol
        if brain_ratio > 0.01:
            if brain_ratio > 0.3:
                brain_slice = binary_fill_holes(
                    binary_closing(brain_slice, iterations=5)
                )
            final_brain[:, :, s] = brain_slice

    time_str = time.strftime("%H:%M:%S")
    print(
        '{:}[{:}]{:} Testing the image{:}'.format(
            c['c'], time_str, c['g'], c['nc'])
    )
    net = NewLesionsAttUNet(n_images=1)
    n_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print('Unet parameters {:d}'.format(n_params))

    case_start = time.time()

    bl = np.expand_dims(get_normalised_image(bl_final, final_brain), axis=0)
    fu = np.expand_dims(get_normalised_image(fu_final, final_brain), axis=0)

    ref_nii = nib.load(fu_final)
    segmentation = np.zeros_like(brain).astype(np.float32)

    brain_bin = final_brain.astype(np.bool)
    idx = np.where(brain_bin)
    bb = tuple(
        slice(min_i, max_i)
        for min_i, max_i in zip(
            np.min(idx, axis=-1), np.max(idx, axis=-1)
        )
    )

    for fi in range(n_folds):
        net_name = 'positive-unet_n{:d}.pt'.format(fi)
        net.load_model(os.path.join(model_path, net_name))
        if verbose > 1:
            print(
                '{:}Running activity pipeline{:} (fold: {:})'.format(
                    c['clr'] + c['c'], c['nc'], c['c'] + str(fi) + c['nc']
                ), end='\r'
            )

        seg = np.zeros(final_brain.shape)
        t_source = bl[(slice(None),) + bb]
        t_target = fu[(slice(None),) + bb]
        seg_bb = net.new_lesions_patch(
            t_source, t_target, patch_size, batch_size,
            0, 1, case_start
        )
        seg[bb] = seg_bb

        seg_temp = np.zeros_like(brain).astype(np.float32)
        seg_temp[bb] = seg_bb
        seg_temp[np.logical_not(brain_bin)] = 0

        segmentation += (seg_temp / n_folds)

    pactivity_name = os.path.join(t_path, activity_name)
    # Thresholding + brain mask filtering
    small_activity = remove_small_regions(
        np.logical_and(segmentation > 0.5, brain_bin), min_size=3
    )
    final_activity = remove_boundary_regions(small_activity, brain_bin)

    # Final mask
    segmentation_nii = nib.Nifti1Image(
        final_activity.astype(np.uint8),
        ref_nii.get_qform()
    )
    segmentation_nii.to_filename(pactivity_name)

    time_str = time.strftime(
        '%H hours %M minutes %S seconds',
        time.gmtime(time.time() - global_start)
    )
    print(
        '{:}Segmentation finished (total time {:})'.format(c['clr'], time_str)
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
