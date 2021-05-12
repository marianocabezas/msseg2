import time
import argparse
import os
from time import strftime
import numpy as np
import nibabel as nib
from nibabel import load as load_nii
from utils import color_codes
from utils import get_mask, get_normalised_image
from utils import time_to_string, find_file
from models import NewLesionsUNet
from utils import remove_small_regions, get_dirs

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
    bl_name = 'flair_time01_on_middle_space_n4.nii.gz'
    fu_name = 'flair_time02_on_middle_space_n4.nii.gz'
    activity_name = 'positive_activity.nii.gz'

    patients = sorted(get_dirs(t_path))

    # Preprocessing
    time_str = time.strftime("%H:%M:%S")
    print(
        '{:}[{:}]{:} Preprocessing the dataset - (path: {:}){:}'.format(
            c['c'], time_str, c['g'], t_path, c['nc'])
    )

    global_start = time.time()
    # Main loop
    for i, patient in enumerate(patients):
        test_elapsed = time.time() - global_start
        test_eta = len(patients) * test_elapsed / (i + 1)
        print(
            '{:}Testing patient {:} ({:d}/{:d}) '
            '{:} ETA {:}'.format(
                c['clr'], patient, i + 1, len(patients),
                time_to_string(test_elapsed),
                time_to_string(test_eta),
            )
        )
        patient_path = os.path.join(t_path, patient)

        case_start = time.time()

        pbrain_name = find_file(brain_name, patient_path)
        brain = get_mask(pbrain_name)
        pbl_name = find_file(bl_name, patient_path)
        pfu_name = find_file(fu_name, patient_path)
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

        pactivity_name = os.path.join(patient_path, activity_name)
        # Thresholding + brain mask filtering
        final_activity = remove_small_regions(
            np.logical_and(segmentation > 0.5, brain.astype(np.bool)),
            min_size=2
        )

        # Final mask
        segmentation_nii = nib.Nifti1Image(
            final_activity, ref_nii.get_qform(), ref_nii.header
        )
        segmentation_nii.to_filename(pactivity_name)

        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - case_start)
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
    print(
        '{:}All patients finished {:}'.format(c['r'], time_str + c['nc'])
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
