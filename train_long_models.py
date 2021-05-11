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
from utils import remove_small_regions
from datasets import LongitudinalCroppingDataset


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
    return vars(parser.parse_args())


"""
> Main functions (options)
"""


def get_data(
        patients,
        d_path=None,
        images=None,
        brain_name='brain_mask.nii.gz',
        positive_name='positive_activity.nii.gz',
):
    if d_path is None:
        d_path = parse_args()['dataset_path']
    if images is None:
        images = ['pd', 't1', 't2', 'flair']
    patient_paths = [
        os.path.join(d_path, centre, patient, 'preprocessed')
        for centre, patient in patients
    ]
    brain_names = [
        os.path.join(p_path, brain_name) for p_path in patient_paths
    ]
    brains = list(map(get_mask, brain_names))

    positive_names = [
        os.path.join(p_path, positive_name) for p_path in patient_paths
    ]
    positive = list(map(get_mask, positive_names))

    norm_source = [
        np.stack(
            [
                get_normalised_image(
                    os.path.join(
                        p, 'bl_{:}_final.nii.gz'.format(im)
                    ),
                    mask_i
                ) for im in images
            ],
            axis=0
        ) for p, mask_i in zip(patient_paths, brains)
    ]

    norm_target = [
        np.stack(
            [
                get_normalised_image(
                    os.path.join(
                        p, 'fu_{:}_final.nii.gz'.format(im)
                    ),
                    mask_i
                ) for im in images
            ],
            axis=0
        ) for p, mask_i in zip(patient_paths, brains)
    ]

    return norm_source, norm_target, positive, brains


def get_patient(
        patient,
        centre,
        d_path=None,
        images=None,
        brain_name='brain_mask.nii.gz',
):
    if d_path is None:
        d_path = parse_args()['dataset_path']
    if images is None:
        images = ['pd', 't1', 't2', 'flair']
    patient_path = os.path.join(d_path, centre, patient, 'preprocessed')
    brain_name = os.path.join(patient_path, brain_name)
    brain = get_mask(brain_name)

    norm_source = np.stack(
        [
            get_normalised_image(
                os.path.join(
                    patient_path, 'bl_{:}_final.nii.gz'.format(im)
                ),
                brain
            ) for im in images
        ],
        axis=0
    )

    norm_target = np.stack(
        [
            get_normalised_image(
                os.path.join(
                    patient_path, 'fu_{:}_final.nii.gz'.format(im)
                ),
                brain
            ) for im in images
        ],
        axis=0
    )

    return norm_source, norm_target, brain


"""
> Main functions (networks)
"""


def train_net(
        net, model_name, train_patients, val_patients, images, epochs=None,
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
        :param images:
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

    try:
        net.load_model(os.path.join(d_path, model_name))
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
        batch_size = parse_args()['batch_size']
        patch_size = parse_args()['patch_size']
        overlap = parse_args()['patch_size'] // 2
        num_workers = 16

        print('Loading the {:}training{:} data'.format(c['b'], c['nc']))
        train_source, train_target, train_masks, train_brains = get_data(
            train_patients, d_path, images=images
        )
        print('Loading the {:}validation{:} data'.format(c['b'], c['nc']))
        val_source, val_target, val_masks, val_brains = get_data(
            val_patients, d_path, images=images
        )

        if verbose > 1:
            print('Training dataset (with validation)')
        train_dataset = LongitudinalCroppingDataset(
            train_source, train_target, train_masks, train_brains,
            patch_size=patch_size, overlap=overlap
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size, True, num_workers=num_workers
        )

        if verbose > 1:
            print('Validation dataset (with validation)')
        val_dataset = LongitudinalCroppingDataset(
            val_source, val_target, val_masks, val_brains,
            patch_size=patch_size, filtered=False
        )
        val_dataloader = DataLoader(
            val_dataset, 4 * batch_size, num_workers=num_workers
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

    net.save_model(os.path.join(d_path, model_name))


def test_net(net, patients, images, filename, gt_name, verbose=1):
    # Init
    c = color_codes()
    d_path = parse_args()['d_path']
    seg_list = list()

    test_start = time.time()
    tests = len(patients)
    for i, (c_i, p) in enumerate(patients):
        test_elapsed = time.time() - test_start
        test_eta = tests * test_elapsed / (i + 1)
        print(
            '{:}Testing patient {:} ({:d}/{:d}) '
            '{:} ETA {:}'.format(
                c['clr'], p, i + 1, len(patients),
                time_to_string(test_elapsed),
                time_to_string(test_eta),
            )
        )
        path = os.path.join(d_path, c_i, p, 'preprocessed')

        if find_file(filename, path) is None:
            seg_name = os.path.join(path, filename)
            nii = load_nii(os.path.join(path, gt_name))
            source, target, brain = get_patient(p, c_i, d_path, images)
            brain_bin = brain.astype(np.bool)
            # Brain mask
            idx = np.where(brain_bin)
            bb = tuple(
                slice(min_i, max_i)
                for min_i, max_i in zip(
                    np.min(idx, axis=-1), np.max(idx, axis=-1)
                )
            )

            try:
                seg = net.new_lesions(source, target)
            except RuntimeError:
                seg = np.zeros(brain.shape)
                seg_bb = net.new_lesions_patch(
                    source[(slice(None),) + bb], target[(slice(None),) + bb],
                    32, 16
                )
                seg[bb] = seg_bb

            seg[np.logical_not(brain_bin)] = 0
            seg_nii = nib.Nifti1Image(
                seg, nii.get_qform(), nii.header
            )
            seg_nii.to_filename(
                os.path.join(path, 'pr_{:}'.format(filename))
            )

            segmentation = remove_small_regions(seg > 0.5, min_size=5)
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
    patients = [
        (c_i, [p for p in get_dirs(os.path.join(d_path, c_i))])
        for c_i in ['GE_3', 'Philips_1.5', 'Philips_3', 'SIEMENS_3']
    ]

    patience = parse_args()['patience']
    epochs = parse_args()['epochs']
    gpu = parse_args()['gpu_id']
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%d' % gpu if cuda else 'cpu')
    images = ['flair']

    for i in range(n_folds):
        if verbose > 0:
            print(
                '{:}Fold {:}{:2d}/{:2d}{:} (n-folds cross-val)'.format(
                    c['clr'] + c['c'], c['g'], i + 1, n_folds, c['nc']
                )
            )
        # Training
        ini_test = [len(p_list) * i // n_folds for c, p_list in patients]
        end_test = [len(p_list) * (i + 1) // n_folds for c, p_list in patients]
        training_set = [
            [(c, p) for p in p_list[end_c:] + p_list[:ini_c]]
            for (c, p_list), ini_c, end_c in zip(patients, ini_test, end_test)
        ]
        val_idx = [
            max(1, int(val_split * len(p_list))) for p_list in training_set
        ]
        val_patients = [
            (c, p) for p_list, idx in zip(training_set, val_idx)
            for (c, p) in p_list[:idx]
        ]
        train_patients = [
            (c, p) for p_list, idx in zip(training_set, val_idx)
            for (c, p) in p_list[idx:]
        ]
        test_patients = [
            (c, p)
            for (c, p_list), ini_c, end_c in zip(patients, ini_test, end_test)
            for p in p_list[ini_c:end_c]
        ]

        print(
            '{:}[{:}]{:} Positive activity {:}Unet{:}'.format(
                c['c'], time.strftime("%H:%M:%S"), c['g'], c['nc'] + c['b'],
                c['nc']
            )
        )
        seg_unet = NewLesionsUNet(device=device, n_images=1)
        model_name = 'positive.unet_n{:d}.e{:d}.p{:d}.mdl'.format(
            i, epochs, patience
        )
        train_net(
            seg_unet, model_name, train_patients, val_patients, images,
            verbose=verbose
        )

        print(
            '{:}[{:}]{:} Positive activity (loo-test){:} {:}Unet{:}'.format(
                c['c'], time.strftime("%H:%M:%S"), c['g'], c['nc'], c['b'],
                c['nc']
            )
        )
        test_net(
            seg_unet, test_patients, images,
            'positive.xval.unet.nii.gz', 'positive_activity.nii.gz',
            verbose=verbose
        )


def main():
    cross_val(verbose=2)


if __name__ == "__main__":
    main()
