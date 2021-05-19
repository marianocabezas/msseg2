import argparse
import os
import time
import numpy as np
import nibabel as nib
from nibabel import load as load_nii
import torch
from torch.utils.data import DataLoader
from utils import get_dirs, time_to_string, find_file
from utils import get_mask, get_normalised_image
from utils import color_codes
from models import NewLesionsUNet
from utils import remove_small_regions, get_bb
from datasets import LongitudinalCroppingDataset, LongitudinalDataset


def parse_args():
    """
    Arguments for the different lesion activity analysis pipelines.
    """
    parser = argparse.ArgumentParser(
        description='Run the longitudinal MS lesion segmentation pipelines.'
    )
    parser.add_argument(
        '-d', '--data-path',
        dest='d_path', default='/data',
        help='Option to use the old pipeline in the production docker. '
             'The second parameter should be the folder where '
             'the patients are stored.'
    )
    parser.add_argument(
        '-m', '--model-directory',
        dest='model_dir', default='./ModelWeightsFinal',
        help='Path to the final model weights.'
    )
    parser.add_argument(
        '-M', '--initial-model-directory',
        dest='init_model_dir', default=None,
        help='Path to the initial model weights (fine-tuning).'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int, default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=5,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-g', '--gpu',
        dest='gpu_id',
        type=int, default=0,
        help='GPU id number'
    )
    parser.add_argument(
        '-B', '--batch-size',
        dest='batch_size',
        type=int, default=32,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-w', '--patch-size',
        dest='patch_size',
        type=int, default=32,
        help='Patch size'
    )
    parser.add_argument(
        '-f', '--freeze-ae',
        dest='freeze_ae', default=False, action='store_true',
        help='Option to freeze the autoencoder.'
    )
    return vars(parser.parse_args())


"""
> Main functions (options)
"""


def get_data(
    patients,
    d_path=None,
    brain_name='brain_mask.nii.gz',
    bl_name='flair_time01_on_middle_space_n4.nii.gz',
    fu_name='flair_time02_on_middle_space_n4.nii.gz',
    positive_name='positive_activity.nii.gz',
):
    if d_path is None:
        d_path = parse_args()['dataset_path']
    patient_paths = [os.path.join(d_path, patient) for patient in patients]
    brain_names = [
        os.path.join(p_path, brain_name) for p_path in patient_paths
    ]
    brains = list(map(get_mask, brain_names))

    positive_names = [
        os.path.join(p_path, positive_name) for p_path in patient_paths
    ]
    positive = list(map(get_mask, positive_names))

    norm_bl = [
        np.expand_dims(
            get_normalised_image(os.path.join(p, bl_name), mask_i), axis=0
        ) for p, mask_i in zip(patient_paths, brains)
    ]

    norm_fu = [
        np.expand_dims(
            get_normalised_image(os.path.join(p, fu_name), mask_i), axis=0
        ) for p, mask_i in zip(patient_paths, brains)
    ]

    return norm_bl, norm_fu, positive, brains


"""
> Main functions (networks)
"""


def train_net(
        net, model_name, train_patients, val_patients, epochs=None,
        patience=None, verbose=1
):
    """
        Function that applies a CNN-based registration approach. The goal of
        this network is to find the atrophy deformation, and how it affects the
        lesion mask, manually segmented on the baseline image.
        :param net:
        :param model_name:
        :param train_patients:
        :param val_patients:
        :param epochs:
        :param patience:
        :param verbose: Verbosity level.
        :return: None.
        """

    c = color_codes()

    # Init
    d_path = parse_args()['d_path']

    if epochs is None:
        epochs = parse_args()['epochs']
    if patience is None:
        patience = parse_args()['patience']

    n_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )

    initial_model = parse_args()['init_model_dir']
    if initial_model is not None:
        initial_weights = os.path.join(initial_model, model_name)
        net.load_model(initial_weights)
        if parse_args()['freeze_ae']:
            net.segmenter[0].freeze()

    model_path = parse_args()['model_dir']
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    try:
        net.load_model(os.path.join(model_path, model_name))
        print(
            '{:}Network loaded{:} ({:d} parameters)'.format(
                c['c'], c['nc'], n_params
            )
        )
    except IOError:
        if verbose > 0:
            print(
                '{:}Starting training{:} ({:d} parameters)'.format(
                    c['c'], c['nc'], n_params
                )
            )

        # Datasets / Dataloaders should be added here
        if verbose > 1:
            print('Preparing the training datasets / dataloaders')
        # batch_size = parse_args()['batch_size']
        # patch_size = parse_args()['patch_size']
        # overlap = parse_args()['patch_size'] // 2
        num_workers = 16

        print('Loading the {:}training{:} data'.format(c['b'], c['nc']))
        train_source, train_target, train_masks, train_brains = get_data(
            train_patients, d_path
        )
        print('Loading the {:}validation{:} data'.format(c['b'], c['nc']))
        val_source, val_target, val_masks, val_brains = get_data(
            val_patients, d_path
        )

        if verbose > 1:
            print('Training dataset (with validation)')
        # train_dataset = LongitudinalCroppingDataset(
        #     train_source, train_target, train_masks, train_brains,
        #     patch_size=patch_size, overlap=overlap
        # )
        # train_dataloader = DataLoader(
        #     train_dataset, batch_size, True, num_workers=num_workers
        # )
        train_dataset = LongitudinalDataset(
            train_source, train_target, train_masks, train_brains
        )
        train_dataloader = DataLoader(
            train_dataset, 1, True, num_workers=num_workers
        )

        if verbose > 1:
            print('Validation dataset (with validation)')
        # val_dataset = LongitudinalCroppingDataset(
        #     val_source, val_target, val_masks, val_brains,
        #     patch_size=patch_size, filtered=False
        # )
        # val_dataloader = DataLoader(
        #     val_dataset, 4 * batch_size, num_workers=num_workers
        # )
        val_dataset = LongitudinalDataset(
            val_source, val_target, val_masks, val_brains
        )
        val_dataloader = DataLoader(
            val_dataset, 1, num_workers=num_workers
        )

        training_start = time.time()

        net.fit(
            train_dataloader,
            val_dataloader,
            epochs=epochs,
            patience=patience
        )

        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - training_start)
            )
            print(
                '{:}Training finished{:} (total time {:})'.format(
                    c['r'], c['nc'], time_str
                )
            )

    net.save_model(os.path.join(model_path, model_name))


def test_net(
    net, patients,
    brain_name='brain_mask.nii.gz',
    bl_name='flair_time01_on_middle_space_n4.nii.gz',
    fu_name='flair_time02_on_middle_space_n4.nii.gz',
    verbose=1
):
    # Init
    c = color_codes()
    d_path = parse_args()['d_path']
    seg_list = list()

    initial_model = parse_args()['init_model_dir']
    fine_tuning = parse_args()['freeze_ae']
    if initial_model is not None:
        if fine_tuning:
            filename = 'positive_activity_ft-freeze.nii.gz'
        else:
            filename = 'positive_activity_ft.nii.gz'
    else:
        filename = 'positive_activity_xval.nii.gz'

    test_start = time.time()
    tests = len(patients)
    for i, p in enumerate(patients):
        test_elapsed = time.time() - test_start
        test_eta = tests * test_elapsed / (i + 1)
        print(
            '{:}Testing patient {:} ({:d}/{:d}) '
            '{:} ETA {:}'.format(
                c['clr'], p, i + 1, len(patients),
                time_to_string(test_elapsed),
                time_to_string(test_eta),
            ),
            end='\r'
        )
        patient_path = os.path.join(d_path, p)

        if find_file(filename, patient_path) is None:
            seg_name = os.path.join(patient_path, filename)
            pbrain_name = find_file(brain_name, patient_path)
            nii = load_nii(pbrain_name)
            brain_bin = nii.get_fdata().astype(bool)
            pbl_name = find_file(bl_name, patient_path)
            pfu_name = find_file(fu_name, patient_path)
            bl = get_normalised_image(pbl_name, brain_bin)
            fu = get_normalised_image(pfu_name, brain_bin)

            # Brain mask
            bb = get_bb(brain_bin, 2)

            seg_bb = net.new_lesions(
                np.expand_dims(bl[bb], axis=0), np.expand_dims(fu[bb], axis=0)
            )
            # try:
            #     seg = net.new_lesions(
            #         np.expand_dims(bl, axis=0), np.expand_dims(fu, axis=0)
            #     )
            # except RuntimeError:
            seg = np.zeros(brain_bin.shape)
            #     seg_bb = net.new_lesions_patch(
            #         np.expand_dims(bl[bb], axis=0),
            #         np.expand_dims(fu[bb], axis=0),
            #         32, 16, i, len(patients), test_start
            #     )
            seg[bb] = seg_bb

            seg[np.logical_not(brain_bin)] = 0
            seg_nii = nib.Nifti1Image(
                seg, nii.get_qform(), nii.header
            )
            seg_nii.to_filename(
                os.path.join(patient_path, 'pr_{:}'.format(filename))
            )

            segmentation = remove_small_regions(seg > 0.5, min_size=3)
            seg_nii = nib.Nifti1Image(
                segmentation, nii.get_qform(), nii.header
            )
            seg_nii.to_filename(seg_name)

            seg_list.append(segmentation)

    if verbose > 0:
        time_str = time.strftime(
            '%H hours %M minutes %S seconds',
            time.gmtime(time.time() - test_start)
        )
        print(
            '{:}Testing finished{:} (total time {:})'.format(
                c['clr'] + c['r'], c['nc'], time_str
            )
        )

    return seg_list


"""
> Main functions (testing everything)
"""


def cross_val(n_folds=5, val_split=0.1, verbose=0):
    # Init
    c = color_codes()
    d_path = parse_args()['d_path']
    patients = sorted(get_dirs(d_path))

    positive_cases = [
        '013', '016', '018', '020', '021', '024', '026', '027', '029', '030',
        '032', '035', '037', '039', '043', '047', '048', '057', '061', '069',
        '074', '077', '083', '088', '091', '094', '095', '099', '100'
    ]
    negative_cases = [
        '015', '019', '049', '051', '052', '068', '070', '084', '089', '090',
        '096'
    ]

    patience = parse_args()['patience']
    epochs = parse_args()['epochs']
    gpu = parse_args()['gpu_id']
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%d' % gpu if cuda else 'cpu')

    for i in range(n_folds):
        if verbose > 0:
            print(
                '{:}Fold {:}{:2d}/{:2d}{:} (n-folds cross-val)'.format(
                    c['clr'] + c['c'], c['g'], i + 1, n_folds, c['nc']
                )
            )
        # > Training cases
        # We will avoid "negative" samples during training.
        # Indices
        ini_pos = len(positive_cases) * i // n_folds
        end_pos = len(positive_cases) * (i + 1) // n_folds
        ini_neg = len(negative_cases) * i // n_folds
        end_neg = len(negative_cases) * (i + 1) // n_folds
        training_set = positive_cases[end_pos:] + positive_cases[:ini_pos]
        val_idx = max(1, int(val_split * len(training_set)))
        val_patients = training_set[:val_idx]
        train_patients = training_set[val_idx:]

        # > Testing cases
        test_pos = positive_cases[ini_pos:end_pos]
        test_neg = negative_cases[ini_neg:end_neg]
        test_patients = test_pos + test_neg

        print(
            '{:}[{:}]{:} Positive activity {:}Unet{:}'.format(
                c['c'], time.strftime("%H:%M:%S"), c['g'], c['nc'] + c['b'],
                c['nc']
            )
        )
        if parse_args()['init_model_dir'] is not None:
            seg_unet = NewLesionsUNet(device=device, n_images=1)
        else:
            seg_unet = NewLesionsUNet(
                device=device, n_images=1, conv_filters=[8, 16, 32, 64, 128, 256]
            )
        model_name = 'positive-unet_n{:d}.pt'.format(
            i, epochs, patience
        )
        train_net(
            seg_unet, model_name, train_patients, val_patients,
            verbose=verbose
        )

        print(
            '{:}[{:}]{:} Positive activity (loo-test){:} {:}Unet{:}'.format(
                c['c'], time.strftime("%H:%M:%S"), c['g'], c['nc'], c['b'],
                c['nc']
            )
        )
        test_net(seg_unet, test_patients, verbose=verbose)


def main():
    cross_val(verbose=2)


if __name__ == "__main__":
    main()
