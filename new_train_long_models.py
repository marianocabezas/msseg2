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
from models import NewLesionsAttUNet, LongitudinalEncoder
from utils import remove_small_regions
from datasets import LongitudinalCroppingDataset
from datasets import LongitudinalImageCroppingDataset


def parse_args():
    """
    Arguments for the different lesion activity analysis pipelines.
    """
    parser = argparse.ArgumentParser(
        description='Run the longitudinal MS lesion segmentation pipelines.'
    )
    parser.add_argument(
        '-d', '--data-path',
        dest='d_path', default=None,
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
        type=int, default=64,
        help='Patch size'
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
    positive_name='ground_truth.nii.gz',
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
        d_path, net, model_name, train_patients, val_patients, epochs=None,
        patience=None, dataset=LongitudinalCroppingDataset, verbose=1
):
    """
        Function that applies a CNN-based registration approach. The goal of
        this network is to find the atrophy deformation, and how it affects the
        lesion mask, manually segmented on the baseline image.
        :param d_path:
        :param net:
        :param model_name:
        :param train_patients:
        :param val_patients:
        :param epochs:
        :param patience:
        :param dataset:
        :param verbose: Verbosity level.
        :return: None.
        """

    # Init
    c = color_codes()

    if epochs is None:
        epochs = parse_args()['epochs']
    if patience is None:
        patience = parse_args()['patience']
    if dataset != LongitudinalCroppingDataset:
        epochs = epochs // 2
        patience = patience // 2
    n_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )

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
        batch_size = parse_args()['batch_size']
        if dataset != LongitudinalCroppingDataset:
            batch_size = batch_size * 4
        patch_size = parse_args()['patch_size']
        overlap = parse_args()['patch_size'] // 2
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
        train_dataset = dataset(
            train_source, train_target, train_masks, train_brains,
            patch_size=patch_size, overlap=overlap
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size, True, num_workers=num_workers
        )

        if verbose > 1:
            print('Validation dataset (with validation)')
        val_dataset = dataset(
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

    net.save_model(os.path.join(model_path, model_name))


def test_net(
    d_path, net, patients,
    brain_name='brain_mask.nii.gz',
    bl_name='flair_time01_on_middle_space_n4.nii.gz',
    fu_name='flair_time02_on_middle_space_n4.nii.gz',
    verbose=1
):
    # Init
    c = color_codes()
    seg_list = list()

    filename = 'positive_activity_final.nii.gz'

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
            # bb = get_bb(brain_bin, 2)

            # seg_bb = net.new_lesions(
            #     np.expand_dims(bl[bb], axis=0), np.expand_dims(fu[bb], axis=0)
            # )
            # try:
            #     seg = net.new_lesions(
            #         np.expand_dims(bl, axis=0), np.expand_dims(fu, axis=0)
            #     )
            # except RuntimeError:
            # seg = np.zeros(brain_bin.shape)
            seg = net.new_lesions_patch(
                # np.expand_dims(bl[bb], axis=0),
                # np.expand_dims(fu[bb], axis=0),
                np.expand_dims(bl, axis=0),
                np.expand_dims(fu, axis=0),
                32, 256, i, len(patients), test_start
            )
            # seg[bb] = seg_bb

            seg[np.logical_not(brain_bin)] = 0
            seg_nii = nib.Nifti1Image(
                seg, nii.get_qform(), nii.header
            )
            seg_nii.to_filename(
                os.path.join(patient_path, 'pr_{:}'.format(filename))
            )

            segmentation = remove_small_regions(seg > 0.5, min_size=4)
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


def private_train(val_split=0.1, verbose=0):
    # Init
    c = color_codes()
    gpu = parse_args()['gpu_id']
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%d' % gpu if cuda else 'cpu')

    d_path = '/data/MSReports/Longitudinal/MICCAI_Challenge2021/private/'
    cases = get_dirs(d_path)
    cases_dict = {}
    for p in cases:
        key = p.split('_')[0]
        print(key)
        try:
            cases_dict[key].append(p)
        except KeyError:
            cases_dict[key] = []
            cases_dict[key].append(p)

    # > Training cases
    val_idx = [
        max(1, int(val_split * len(train_i)))
        for train_i in cases_dict
    ]
    train_patients = [
        p for train_i, idx_i in zip(cases_dict, val_idx)
        for p in train_i[idx_i:]
    ]
    val_patients = [
        p for train_i, idx_i in zip(cases_dict, val_idx)
        for p in train_i[:idx_i]
    ]

    pretrain_net = LongitudinalEncoder(device=device, n_images=1)

    print(
        '{:}[{:}]{:} Positive activity {:}Unet {:}(private){:}'.format(
            c['c'], time.strftime("%H:%M:%S"), c['g'], c['nc'] + c['b'],
            c['y'], c['nc']
        )
    )

    train_net(
        d_path, pretrain_net, 'encoder-net.pt', train_patients, val_patients,
        dataset=LongitudinalImageCroppingDataset, verbose=verbose
    )

    seg_net = NewLesionsAttUNet(device=device, n_images=1)
    seg_net.ae.up = pretrain_net.ae.up
    for param in seg_net.ae.up.parameters():
        param.requires_grad = False
    train_net(
        d_path, seg_net, 'positive-unet.pt', train_patients, val_patients,
        verbose=verbose
    )


def cross_val(n_folds=5, val_split=0.1, verbose=0):
    # Init
    c = color_codes()
    d_path = '/data/MSReports/Longitudinal/MICCAI_Challenge2021/training/'
    positive_cases = [
        '013', '016', '018', '020', '021', '024', '026', '027', '029', '030',
        '032', '035', '037', '039', '043', '047', '048', '057', '061', '069',
        '074', '077', '083', '088', '091', '094', '095', '099', '100'
    ]
    negative_cases = [
        '015', '019', '049', '051', '052', '068', '070', '084', '089', '090',
        '096'
    ]

    model_path = parse_args()['model_dir']
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
        # Indices
        ini_pos = len(positive_cases) * i // n_folds
        end_pos = len(positive_cases) * (i + 1) // n_folds
        ini_neg = len(negative_cases) * i // n_folds
        end_neg = len(negative_cases) * (i + 1) // n_folds
        training_set = positive_cases[end_pos:] + positive_cases[:ini_pos]

        val_idx = max(1, int(val_split * len(training_set)))
        train_patients = training_set[val_idx:]
        val_patients = training_set[:val_idx]

        # > Testing cases
        test_pos = positive_cases[ini_pos:end_pos]
        test_neg = negative_cases[ini_neg:end_neg]
        test_patients = test_pos + test_neg

        print(
            '{:}[{:}]{:} Positive activity {:}Unet {:}(attention){:}'.format(
                c['c'], time.strftime("%H:%M:%S"), c['g'], c['nc'] + c['b'],
                c['y'], c['nc']
            )
        )

        seg_net = NewLesionsAttUNet(device=device, n_images=1)
        seg_net.load_model(os.path.join(model_path, 'positive-unet.pt'))
        for param in seg_net.ae.up.parameters():
            param.requires_grad = False
        model_name = 'positive-unet_n{:d}.pt'.format(i)
        train_net(
            d_path, seg_net, model_name, train_patients, val_patients,
            verbose=verbose
        )

        print(
            '{:}[{:}]{:} Positive activity (loo-test){:} {:}Unet{:}'.format(
                c['c'], time.strftime("%H:%M:%S"), c['g'], c['nc'], c['b'],
                c['nc']
            )
        )

        test_net(d_path, seg_net, test_patients, verbose=verbose)


def main():
    private_train(verbose=2)
    cross_val(verbose=2)


if __name__ == "__main__":
    main()
