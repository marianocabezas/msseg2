import argparse
import os
import time
from functools import reduce
import numpy as np
import SimpleITK as itk
import nibabel as nib
from nibabel import load as load_nii
from nibabel import save as save_nii
from skimage.measure import label as bwlabeln
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_closing as imclose
from scipy.ndimage.morphology import binary_erosion as imerode
from scipy.ndimage.morphology import binary_dilation as imdilate
from scipy.ndimage.morphology import binary_fill_holes
from utils import find_file, get_dirs, time_f
from utils import color_codes, print_message


def parse_args():
    """
    Arguments for the different lesion activity analysis pipelines.
    """
    parser = argparse.ArgumentParser(
        description='Run the longitudinal MS lesion preprocessing pipeline.'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-d', '--data-path',
        dest='d_path', default='/home/mariano/data/longitudinal/5centres',
        help='Option to use the old pipeline in the production docker. '
             'The second parameter should be the folder where '
             'the patients are stored.'
    )
    return vars(parser.parse_args())


"""
> Main functions (preprocessing)
"""


def itkn4(
        image,
        path=None,
        name=None,
        mask=None,
        max_iters=400,
        levels=3,
        cast=itk.sitkFloat32,
        verbose=1
):
    """

    :param image:
    :param path:
    :param name:
    :param mask:
    :param max_iters:
    :param levels:
    :param cast:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(image, str):
        image = itk.ReadImage(image)
    elif isinstance(image, np.ndarray):
        image = itk.GetImageFromArray(image)

    if verbose > 1:
        print('-> Image: ' + os.path.join(path, name + '_corrected.nii.gz'))
    found = find_file(name + '_corrected.nii.gz', path)
    if path is None or name is None or found is None:
        if mask is not None:
            if isinstance(mask, str):
                mask = itk.ReadImage(mask)
            elif isinstance(mask, np.ndarray):
                mask = itk.GetImageFromArray(mask)
        else:
            mask = itk.OtsuThreshold(image, 0, 1, 200)
        image = itk.Cast(image, cast)
        corrector = itk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([max_iters] * levels)
        output = corrector.Execute(image, mask)
        if name is not None and path is not None:
            itk.WriteImage(
                output, os.path.join(path, name + '_corrected.nii.gz')
            )
        return itk.GetArrayFromImage(output)


def wm_segmentation(
        main_dir,
        preprocessed_dir='preprocessed',
        segmentation_dir='segmentation',
        mask_name='brain_mask.nii.gz',
        th=0.5,
        verbose=1,
):
    """
    :param main_dir: Main directory where all patients are stored. Patients
     should be stored on separate folders.
    :param preprocessed_dir: Name of the directory where images are stored.
     The defaults are coherent with all the other itk tools.
    :param segmentation_dir: Name of the directory where tissue segmentations
     will be stored. The defaults are coherent with all the other itk tools.
    :param mask_name: Filename that contains the brain mask image.
    :param th: Threshold for the tissue segmentation step. This threshold is
     used for the trimmed likelihood estimator during the expectation
     maximisation approach. Unlike previous C++ versions of this method, this
     threshold is adaptive to allow the mean and covariance computation even
     with values under this threshold for each class.
    :param verbose: Verbose levels for this tool. The minimum value must be 1.
     For this level of verbosity, only "required" messages involving each step
     and likelihood will be shown. For the next level, various debugging
     options related to the expectation maximisation will be shown.
    :return: None.
    """
    if verbose > 0:
        print('/-------------------------------\\')
        print('|             WMtool            |')
        print('\\-------------------------------/')

    if verbose > 0:
        print('\t-------------------------------')
        print('\t         Segmentation          ')
        print('\t-------------------------------')
        print('\t\\- WM mask segmentation')
    segmentation_path = os.path.join(main_dir, segmentation_dir)
    if not os.path.isdir(segmentation_path):
        os.mkdir(segmentation_path)

    image_path = os.path.join(main_dir, preprocessed_dir)
    wmmask_path = os.path.join(
        segmentation_path,
        'wmmask.nii.gz'
    )
    assert th <= 1, 'Wrong threshold'

    if find_file('wmmask.nii.gz', segmentation_path) is None:
        mask_nii = load_nii(
            os.path.join(
                main_dir,
                mask_name,
            )
        )
        mask = mask_nii.get_data()

        def get_wm(name, maxim=False, bins=128):
            if verbose > 1:
                max_s = 'maximise' if maxim else 'minimise'
                print('\t\t %s (%s)' % (name, max_s))
            image = load_nii(name).get_data()
            # We'll need to compute the cumulative histogram first.
            h, b = np.histogram(image[mask > 0], bins=bins)
            h = h / float(sum(h))
            h_sum = np.cumsum(h) if maxim else np.cumsum(h[::-1])[::-1]
            # Depending on the image, the WM is the most or the least intense
            # tissue.
            h_wm = np.where(h_sum > th)
            t = b[np.min(h_wm)] if maxim else b[np.max(h_wm)]
            if verbose > 1:
                print('\t\t- Threshold %f' % t)
            return (image > t if maxim else image < t) * mask

        image_names = map(
            lambda im: find_file(im + '_processed.nii.gz', image_path),
            ['t1', 't2', 'pd']
        )
        name_maxim = map(
            lambda im_y: (
                os.path.join(image_path, im_y[0]),
                im_y[1]
            ),
            filter(
                lambda x_y: x_y[0] is not None,
                zip(image_names, [True, False, False])
            )
        )

        wm_masks = map(
            lambda im_max: get_wm(im_max[0], im_max[1]),
            name_maxim
        )
        mask_nii.get_data()[:] = reduce(np.logical_and, wm_masks)
        save_nii(mask_nii, wmmask_path)


"""
> Main functions (preprocessing)
"""


def subtraction(followup_name, baseline_name, mask, path):
    # First we'll compute the plain old subtractions.
    # Then we'll apply masks and smoothing.
    followup = load_nii(followup_name)
    baseline = load_nii(baseline_name)
    sub_name = os.path.join(
        path, 'subtraction.nii.gz'
    )
    sub_nii = nib.Nifti1Image(
        followup.get_fdata() - baseline.get_fdata(),
        followup.get_qform(),
        followup.header
    )
    sub_nii.to_filename(sub_name)
    sub = itk.ReadImage(sub_name)

    # Now we mask first with the brainmask...
    sub_name = 'brain_subtraction.nii.gz'
    brainmask = itk.ReadImage(mask)
    brainmask.SetDirection(sub.GetDirection())
    brainmask.CopyInformation(sub)
    brain_sub = itk.Mask(sub, itk.Cast(brainmask, itk.sitkUInt8))
    itk.WriteImage(brain_sub, os.path.join(path, sub_name))


def main():
    # Init
    c = color_codes()
    config = parse_args()
    d_path = config['d_path']

    patients = sorted(get_dirs(d_path))

    # Preprocessing
    time_str = time.strftime("%H:%M:%S")
    print(
        '{:}[{:}]{:} Preprocessing the dataset - (path: {:}){:}'.format(
            c['c'], time_str, c['g'], d_path, c['nc'])
    )

    global_start = time.time()
    # Main loop
    for i, patient in enumerate(patients):
        patient_path = os.path.join(d_path, patient)
        patient_start = time.time()
        print(
            '{:}[{:}]{:} Starting preprocessing with patient {:} {:}'
            '({:3d}/{:3d}){:}'.format(
                c['c'], time.strftime("%H:%M:%S"), c['g'], patient,
                c['c'], i + 1, len(patients), c['nc']
            )
        )
        mask_name = os.path.join(patient_path, 'brain_mask.nii.gz')

        bl_n4_name = find_file(
            'flair_time01_on_middle_space_n4', patient_path
        )
        fu_n4_name = find_file(
            'flair_time02_on_middle_space_n4', patient_path
        )
        if bl_n4_name is None:
            bl_n4_name = os.path.join(
                patient_path, 'flair_time01_on_middle_space_n4.nii.gz'
            )
            bl_name = os.path.join(
                patient_path, 'flair_time01_on_middle_space.nii.gz'
            )
            corr_name = os.path.join(
                patient_path, 'bl_corrected.nii.gz'
            )
            print('\\- Correcting FLAIR BL')
            time_f(
                lambda: itkn4(
                    bl_name,
                    patient_path, 'bl', verbose=2
                ),
            )
            os.rename(corr_name, bl_n4_name)

        if fu_n4_name is None:
            fu_n4_name = os.path.join(
                patient_path, 'flair_time02_on_middle_space_n4.nii.gz'
            )
            fu_name = os.path.join(
                patient_path, 'flair_time02_on_middle_space.nii.gz'
            )
            corr_name = os.path.join(
                patient_path, 'fu_corrected.nii.gz'
            )
            print('\\- Correcting FLAIR FU')
            time_f(
                lambda: itkn4(
                    fu_name,
                    patient_path, 'fu', verbose=2
                ),
            )
            os.rename(corr_name, fu_n4_name)

        if find_file('newbrain_mask.nii.gz', patient_path) is None:
            fu_nii = load_nii(fu_n4_name)
            bl_nii = load_nii(bl_n4_name)

            fu = fu_nii.get_fdata()
            qform = fu_nii.get_qform()
            header = fu_nii.header
            bl = bl_nii.get_fdata()
            fu_mask = fu > threshold_otsu(fu)
            bl_mask = bl > threshold_otsu(bl)

            mask = imerode(
                np.logical_and(bl_mask, fu_mask), iterations=5
            )
            mask_cc = bwlabeln(mask)
            mask_lab = np.argmax(np.bincount(mask_cc.flat)[1:]) + 1
            mask = imdilate(mask_cc == mask_lab, iterations=5)
            brain = binary_fill_holes(imclose(mask, iterations=20))
            final_brain = np.zeros_like(brain)
            slice_vol = brain.shape[0] * brain.shape[1]
            for s in range(brain.shape[-1]):
                brain_slice = brain[:, :, s]
                brain_ratio = np.sum(brain_slice) / slice_vol
                if brain_ratio > 0.01:
                    if brain_ratio > 0.3:
                        brain_slice = binary_fill_holes(
                            imclose(brain_slice, iterations=5)
                        )
                    final_brain[:, :, s] = brain_slice
            brain_nii = nib.Nifti1Image(final_brain, qform, header)
            brain_nii.to_filename(mask_name)

            ''' Subtraction '''
            print_message(
                'Subtraction - {:}'.format(patient)
            )

            time_f(
                lambda: subtraction(
                    os.path.join(
                        patient_path,
                        'flair_time02_on_middle_space_n4.nii.gz'
                    ),
                    os.path.join(
                        patient_path,
                        'flair_time01_on_middle_space_n4.nii.gz'
                    ),
                    mask_name, patient_path
                )
            )

            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - patient_start)
            )
            print(
                '{:}Patient {:} finished{:} (total time {:})\n'.format(
                    c['r'], patient, c['nc'], time_str
                )
            )

    time_str = time.strftime(
        '%H hours %M minutes %S seconds',
        time.gmtime(time.time() - global_start)
    )
    print_message(
        '{:}All patients finished {:}'.format(c['r'], time_str + c['nc'])
    )


if __name__ == "__main__":
    main()
