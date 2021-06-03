import time
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel, ResConv3dBlock
from base import Autoencoder, DualAttentionAutoencoder
from utils import time_to_string, to_torch_var
from criteria import gendsc_loss, new_loss, similarity_loss
from criteria import tp_binary_loss, tn_binary_loss, dsc_binary_loss


def norm_f(n_f):
    return nn.GroupNorm(n_f // 4, n_f)


def print_batch(pi, n_patches, i, n_cases, t_in, t_case_in):
    init_c = '\033[38;5;238m'
    percent = 25 * (pi + 1) // n_patches
    progress_s = ''.join(['█'] * percent)
    remainder_s = ''.join([' '] * (25 - percent))

    t_out = time.time() - t_in
    t_case_out = time.time() - t_case_in
    time_s = time_to_string(t_out)

    t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
    eta_s = time_to_string(t_eta)
    pre_s = '{:}Case {:03d}/{:03d} ({:03d}/{:03d} - {:06.2f}%) [{:}{:}]' \
            ' {:} ETA: {:}'
    batch_s = pre_s.format(
        init_c, i + 1, n_cases, pi + 1, n_patches, 100 * (pi + 1) / n_patches,
        progress_s, remainder_s, time_s, eta_s + '\033[0m'
    )
    print('\033[K', end='', flush=True)
    print(batch_s, end='\r', flush=True)


class NewLesionsUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super(NewLesionsUNet, self).__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.segmenter = nn.Sequential(
            Autoencoder(
                self.conv_filters, device, 2 * n_images, block=ResConv3dBlock,
                norm=norm_f
            ),
            nn.Conv3d(self.conv_filters[0], 1, 1)
        )
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'pdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t)
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, source, target):
        data = torch.cat([source, target], dim=1)

        return torch.sigmoid(self.segmenter(data))

    def new_lesions(self, source, target):
        # Init
        self.eval()

        with torch.no_grad():
            seg = self(
                to_torch_var(
                    np.expand_dims(source, axis=0), self.device
                ),
                to_torch_var(
                    np.expand_dims(target, axis=0), self.device
                )
            )
            torch.cuda.empty_cache()

        return seg[0, 0].cpu().numpy()

    def new_lesions_patch(
        self, source, target, patch_size, batch_size,
        case=0, n_cases=1, t_start=None
    ):
        # Init
        self.eval()

        # Init
        t_in = time.time()
        if t_start is None:
            t_start = t_in

        # This branch is only used when images are too big. In this case
        # they are split in patches and each patch is trained separately.
        # Currently, the image is partitioned in blocks with no overlap,
        # however, it might be a good idea to sample all possible patches,
        # test them, and average the results. I know both approaches
        # produce unwanted artifacts, so I don't know.
        # Initial results. Filled to 0.
        seg = np.zeros(source.shape[1:])
        counts = np.zeros(source.shape[1:])

        # The following lines are just a complicated way of finding all
        # the possible combinations of patch indices.
        steps = [
            list(
                range(0, lim - patch_size, patch_size // 4)
            ) + [lim - patch_size]
            for lim in source.shape[1:]
        ]

        steps_product = list(itertools.product(*steps))
        batches = range(0, len(steps_product), batch_size)
        n_batches = len(batches)

        # The following code is just a normal test loop with all the
        # previously computed patches.
        for bi, batch in enumerate(batches):
            # Here we just take the current patch defined by its slice
            # in the x and y axes. Then we convert it into a torch
            # tensor for testing.
            slices = [
                (
                    slice(xi, xi + patch_size),
                    slice(xj, xj + patch_size),
                    slice(xk, xk + patch_size)
                )
                for xi, xj, xk in steps_product[batch:(batch + batch_size)]
            ]

            source_batch = [
                source[slice(None), xslice, yslice, zslice]
                for xslice, yslice, zslice in slices
            ]
            target_batch = [
                target[slice(None), xslice, yslice, zslice]
                for xslice, yslice, zslice in slices
            ]

            # Testing itself.
            with torch.no_grad():
                target_tensor = to_torch_var(np.stack(target_batch, axis=0))
                source_tensor = to_torch_var(np.stack(source_batch, axis=0))
                seg_out = self(source_tensor, target_tensor)
                torch.cuda.empty_cache()

            # Then we just fill the results image.
            for si, (xslice, yslice, zslice) in enumerate(slices):
                counts[xslice, yslice, zslice] += 1
                seg_bi = seg_out[si, 0].cpu().numpy()
                seg[xslice, yslice, zslice] += seg_bi

            # Printing
            print_batch(bi, n_batches, case, n_cases, t_start, t_in)

        seg /= counts

        return seg


class NewLesionsAttUNet(NewLesionsUNet):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super(NewLesionsAttUNet, self).__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [16, 32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.ae = DualAttentionAutoencoder(
            self.conv_filters, device, 2 * n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.ae.to(device)
        self.segmenter = nn.Conv3d(self.conv_filters[0], 1, 1)
        self.segmenter.to(device)

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, source, target):
        features = self.ae(source, target)
        return torch.sigmoid(self.segmenter(features))


class LongitudinalEncoder(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=1,
            dropout=0,
            verbose=0,
    ):
        super(LongitudinalEncoder, self).__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [16, 32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.ae = Autoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.ae.to(device)
        self.final = ResConv3dBlock(
            self.conv_filters[0], 1, 1, norm_f, nn.Identity
        )
        self.final.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'bl',
                'weight': 1,
                'f': lambda p, t: F.mse_loss(
                    p[0], t[0],
                )
            },
            {
                'name': 'fu',
                'weight': 1,
                'f': lambda p, t: F.mse_loss(
                    p[1], t[1],
                )
            },
            {
                'name': 'sim',
                'weight': 1,
                'f': lambda p, t: similarity_loss(p[2])
            },
        ]
        self.val_functions = self.train_functions

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, source, target):
        data = torch.cat([source, target], dim=1)

        return torch.sigmoid(self.segmenter(data))
