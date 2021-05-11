import SimpleITK as SItk
import os
import numpy as np
from .utils import find_file


def print_current(reg_method, tf):
    """

    :param reg_method:
    :param tf:
    :return:
    """
    print(
        '\033[KMI ({:d}): {:5.3f} - {:}: [{:}]'.format(
            reg_method.GetOptimizerIteration(),
            reg_method.GetMetricValue(),
            tf.GetName(),
            ', '.join(['{:5.3f}'.format(p) for p in tf.GetParameters()])
        ), end='\r'
    )


def itkresample(
        fixed,
        moving,
        transform=None,
        path=None,
        name=None,
        default_value=0.0,
        interpolation=SItk.sitkBSpline,
        verbose=0
):
    """

    :param fixed:
    :param moving:
    :param transform:
    :param path:
    :param name:
    :param default_value:
    :param interpolation:
    :param verbose:
    :return:
    """

    interpolation_dict = {
        'linear': SItk.sitkLinear,
        'bspline': SItk.sitkBSpline,
        'nn': SItk.sitkNearestNeighbor,
    }

    # Init
    if isinstance(fixed, str):
        fixed = SItk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = SItk.GetImageFromArray(fixed)
    if isinstance(moving, str):
        moving = SItk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = SItk.GetImageFromArray(moving)

    if verbose > 1:
        print('-> Image: ' + os.path.join(path, name + '.nii.gz'))

    file = find_file(name + '.nii.gz', path)
    if path is None or name is None or file is None:
        interp_alg = interpolation if not isinstance(interpolation, str)\
            else interpolation_dict[interpolation]
        resample = SItk.ResampleImageFilter()
        resample.SetInterpolator(interp_alg)
        resample.SetOutputDirection(fixed.GetDirection())
        resample.SetOutputOrigin(fixed.GetOrigin())
        resample.SetOutputSpacing(fixed.GetSpacing())
        resample.SetSize(fixed.GetSize())

        if transform is not None:
            resample.SetTransform(transform)

        mov_size = moving.GetSize()
        if len(mov_size) == 4:
            extractor = SItk.ExtractImageFilter()
            extractor.SetSize(mov_size[:-1] + (0,))

            images = []
            for idx in range(mov_size[3]):
                extractor.SetIndex([0, 0, 0, idx])
                image = extractor.Execute(moving)
                images.append(resample.Execute(image))

            resampled = SItk.JoinSeries(images)
        else:
            resampled = resample.Execute(moving)

        if path is not None and name is not None:
            SItk.WriteImage(resampled, os.path.join(path, name + '.nii.gz'))
    else:
        resampled = SItk.ReadImage(file)

    return SItk.GetArrayFromImage(resampled)


def itkwarp(
        fixed,
        moving,
        field,
        path=None,
        name=None,
        default_value=0.0,
        interpolation=SItk.sitkBSpline,
        verbose=0,
):
    """

    :param fixed: Fixed image
    :param moving: Moving image
    :param field: Displacement field
    :param path:
    :param name:
    :param default_value:
    :param interpolation: interpolation function
    :param verbose:
    :return:
    """

    if isinstance(field, str):
        field = SItk.ReadImage(field)

    df_transform = SItk.DisplacementFieldTransform(field)

    return itkresample(
        fixed, moving, df_transform,
        path, name, default_value, interpolation, verbose
    )


def itkn4(
        image,
        path=None,
        name=None,
        mask=None,
        max_iters=400,
        levels=3,
        cast=SItk.sitkFloat32,
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
        image = SItk.ReadImage(image)
    elif isinstance(image, np.ndarray):
        image = SItk.GetImageFromArray(image)

    if verbose > 1:
        print('-> Image: ' + os.path.join(path, name + '_corrected.nii.gz'))
    found = find_file(name + '_corrected.nii.gz', path)
    if path is None or name is None or found is None:
        if mask is not None:
            if isinstance(mask, str):
                mask = SItk.ReadImage(mask)
            elif isinstance(mask, np.ndarray):
                mask = SItk.GetImageFromArray(mask)
        else:
            mask = SItk.OtsuThreshold(image, 0, 1, 200)
        image = SItk.Cast(image, cast)
        corrector = SItk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([max_iters] * levels)
        output = corrector.Execute(image, mask)
        if name is not None and path is not None:
            SItk.WriteImage(
                output, os.path.join(path, name + '_corrected.nii.gz')
            )
        return SItk.GetArrayFromImage(output)


def itkhist_match(
        fixed,
        moving,
        path=None,
        name=None,
        histogram_levels=1024,
        match_points=7,
        mean_on=True,
        verbose=1
):
    """

    :param fixed:
    :param moving:
    :param path:
    :param name:
    :param histogram_levels:
    :param match_points:
    :param mean_on:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(fixed, str):
        fixed = SItk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = SItk.GetImageFromArray(fixed)
    if isinstance(moving, str):
        moving = SItk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = SItk.GetImageFromArray(moving)

    if verbose > 1:
        print('-> Image: ' + os.path.join(path, name + '_corrected_matched.nii.gz'))
    if path is None or name is None or find_file(name + '_corrected_matched.nii.gz', path) is None:
        matched = SItk.HistogramMatching(
            SItk.Cast(moving, fixed.GetPixelID()), fixed,
            histogram_levels, match_points, mean_on
        )
        if name is not None and path is not None:
            SItk.WriteImage(matched, os.path.join(path, name + '_corrected_matched.nii.gz'))
        return SItk.GetArrayFromImage(matched)


def itksmoothing(image, path=None, name=None, sigma=0.5, sufix='_smoothed_subtraction.nii.gz', verbose=1):
    """

    :param image:
    :param path:
    :param name:
    :param sigma:
    :param sufix:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(image, str):
        image = SItk.ReadImage(image)
    elif isinstance(image, np.ndarray):
        image = SItk.GetImageFromArray(image)

    # Gaussian smoothing
    gauss_filter = SItk.DiscreteGaussianImageFilter()
    gauss_filter.SetVariance(sigma * sigma)

    if verbose > 1:
        print('-> Image: ' + os.path.join(path, name + sufix))
    if path is None or name is None or find_file(name + sufix, path) is None:
        smoothed = gauss_filter.Execute(image)
        SItk.WriteImage(smoothed, os.path.join(path, name + sufix))
    else:
        smoothed = SItk.ReadImage(os.path.join(path, name + sufix))
    return SItk.GetArrayFromImage(smoothed)


def itkrigid(
        fixed,
        moving,
        name='',
        fixed_mask=None,
        moving_mask=None,
        number_bins=50,
        levels=3,
        steps=50,
        sampling=0.5,
        learning_rate=1.0,
        min_step=0.0001,
        max_step=0.2,
        relaxation_factor=0.5,
        cast=SItk.sitkFloat32,
        verbose=1
):
    """

    :param fixed:
    :param moving:
    :param name:
    :param fixed_mask:
    :param moving_mask:
    :param number_bins:
    :param levels:
    :param steps:
    :param sampling:
    :param learning_rate:
    :param min_step:
    :param max_step:
    :param relaxation_factor:
    :param cast:
    :param verbose:
    :return:
    """
    # Init
    if isinstance(fixed, str):
        fixed = SItk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = SItk.GetImageFromArray(fixed)
    if isinstance(moving, str):
        moving = SItk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = SItk.GetImageFromArray(moving)
    fixed_float32 = SItk.Cast(fixed, cast)
    moving_float32 = SItk.Cast(moving, cast)

    ''' Transformations '''
    initial_tf = SItk.CenteredTransformInitializer(
        fixed_float32,
        moving_float32,
        SItk.Euler3DTransform(),
        SItk.CenteredTransformInitializerFilter.MOMENTS
    )

    ''' Registration parameters '''
    if verbose > 0:
        print('Rigid initial registration')
    registration = SItk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_bins)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling)
    registration.SetInterpolator(SItk.sitkLinear)

    # Masks
    if fixed_mask is not None:
        if isinstance(fixed_mask, str):
            fixed_mask = SItk.ReadImage(fixed_mask)
        elif isinstance(fixed_mask, np.ndarray):
            fixed_mask = SItk.GetImageFromArray(fixed_mask)
        registration.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        if isinstance(moving_mask, str):
            moving_mask = SItk.ReadImage(moving_mask)
        elif isinstance(moving_mask, np.ndarray):
            moving_mask = SItk.GetImageFromArray(moving_mask)
        registration.SetMetricMovingMask(moving_mask)

    # Optimizer settings.
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=steps,
        maximumStepSizeInPhysicalUnits=max_step,
        relaxationFactor=relaxation_factor
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    smoothing_sigmas = range(levels - 1, -1, -1)
    if verbose > 1:
        print('> Sigmas {:}'.format(smoothing_sigmas))
    shrink_factor = [2**i for i in smoothing_sigmas]
    registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factor)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Connect all of the observers so that we can perform plotting during registration.
    if verbose > 0:
        registration.AddCommand(
            SItk.sitkMultiResolutionIterationEvent,
            lambda: print('\033[K> {:} ({:}) level {:d}'.format(
                registration.GetName(),
                name,
                registration.GetCurrentLevel()
            ))
        )
    if verbose > 1:
        registration.AddCommand(
            SItk.sitkIterationEvent,
            lambda: print_current(registration, initial_tf)
        )

    # Initial versor optimisation
    registration.SetInitialTransform(initial_tf)
    registration.Execute(fixed_float32, moving_float32)
    if verbose > 0:
        print('\033[KRegistration finished')

    return initial_tf


def itkaffine(
        fixed,
        moving,
        name='',
        fixed_mask=None,
        moving_mask=None,
        initial_tf=None,
        number_bins=50,
        levels=3,
        steps=50,
        sampling=0.5,
        learning_rate=1.0,
        min_step=0.0001,
        max_step=0.2,
        relaxation_factor=0.5,
        cast=SItk.sitkFloat32,
        verbose=1
):
    """

    :param fixed:
    :param moving:
    :param name:
    :param fixed_mask:
    :param moving_mask:
    :param initial_tf:
    :param number_bins:
    :param levels:
    :param steps:
    :param sampling:
    :param learning_rate:
    :param min_step:
    :param max_step:
    :param relaxation_factor:
    :param cast:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(fixed, str):
        fixed = SItk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = SItk.GetImageFromArray(fixed)
    if isinstance(moving, str):
        moving = SItk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = SItk.GetImageFromArray(moving)
    if initial_tf is None:
        initial_tf = itkrigid(
            fixed, moving, name, fixed_mask, moving_mask, verbose=verbose
        )

    if verbose > 0:
        print('Affine registration')
    fixed_float32 = SItk.Cast(fixed, cast)
    moving_float32 = SItk.Cast(moving, cast)
    optimized_tf = SItk.AffineTransform(3)

    ''' Registration parameters '''
    registration = SItk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_bins)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling)
    registration.SetInterpolator(SItk.sitkLinear)
    if fixed_mask is not None:
        if isinstance(fixed_mask, str):
            fixed_mask = SItk.ReadImage(fixed_mask)
        elif isinstance(fixed_mask, np.ndarray):
            fixed_mask = SItk.GetImageFromArray(fixed_mask)
        registration.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        if isinstance(moving_mask, str):
            moving_mask = SItk.ReadImage(moving_mask)
        elif isinstance(moving_mask, np.ndarray):
            moving_mask = SItk.GetImageFromArray(moving_mask)
        registration.SetMetricMovingMask(moving_mask)

    # Optimizer settings.
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=steps,
        maximumStepSizeInPhysicalUnits=max_step,
        relaxationFactor=relaxation_factor
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    smoothing_sigmas = range(levels - 1, -1, -1)
    if verbose > 1:
        print('> Sigmas {:}'.format(smoothing_sigmas))
    shrink_factor = [2**i for i in smoothing_sigmas]
    registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factor)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    '''Affine'''
    # Optimizer settings.
    registration.RemoveAllCommands()
    if verbose > 0:
        registration.AddCommand(
            SItk.sitkMultiResolutionIterationEvent,
            lambda: print('\033[K> {:} ({:}) level {:d}'.format(
                registration.GetName(),
                name,
                registration.GetCurrentLevel()
            ))
        )
    if verbose > 1:
        registration.AddCommand(
            SItk.sitkIterationEvent,
            lambda: print_current(registration, optimized_tf)
        )

    registration.SetMovingInitialTransform(initial_tf)
    registration.SetInitialTransform(optimized_tf)

    registration.Execute(fixed_float32, moving_float32)
    if verbose > 0:
        print('\033[KRegistration finished')

    final_tf = SItk.Transform(optimized_tf)
    final_tf.AddTransform(initial_tf)

    return final_tf


def itksubtraction(fixed, moving, path=None, name=None, verbose=1):
    """

    :param fixed:
    :param moving:
    :param path:
    :param name:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(fixed, str):
        fixed = SItk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = SItk.GetImageFromArray(fixed)
    if isinstance(moving, str):
        moving = SItk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = SItk.GetImageFromArray(moving)

    if verbose > 1:
        print('-> Image: ' + os.path.join(path, name + '_subtraction.nii.gz'))

    if path is None or name is None or find_file(name + '_subtraction.nii.gz', path) is None:
        sub = SItk.Subtract(
            SItk.Cast(fixed, SItk.sitkFloat32),
            SItk.Cast(moving, SItk.sitkFloat32)
        )
        if path is not None and name is not None:
            SItk.WriteImage(sub, os.path.join(path, name + '_subtraction.nii.gz'))
    else:
        sub = SItk.ReadImage(os.path.join(path, name + '_subtraction.nii.gz'))

    return SItk.GetArrayFromImage(sub)


def itkdemons(
        fixed,
        moving,
        mask=None,
        path=None,
        name=None,
        steps=50,
        sigma=1.0,
        cast=SItk.sitkFloat32,
        verbose=1
):
    """

    :param fixed:
    :param moving:
    :param mask:
    :param path:
    :param name:
    :param steps:
    :param sigma:
    :param cast:
    :param verbose:
    :return:
    """
    # Init
    if isinstance(fixed, str):
        fixed = SItk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = SItk.GetImageFromArray(fixed)
    if isinstance(moving, str):
        moving = SItk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = SItk.GetImageFromArray(moving)
    if mask is not None:
        if isinstance(mask, str):
            mask = SItk.ReadImage(mask)
        elif isinstance(mask, np.ndarray):
            mask = SItk.GetImageFromArray(mask)
        fixed = SItk.Mask(fixed, mask)
        moving = SItk.Mask(moving, mask)

    fixed_float32 = SItk.Cast(fixed, cast)
    moving_float32 = SItk.Cast(moving, cast)

    if name is not None:
        deformation_name = name + '_multidemons_deformation.nii.gz'
    else:
        deformation_name = '/null'

    if verbose > 1:
        print('-> Deformation: ' + os.path.join(path, deformation_name))

    if path is None or name is None or find_file(deformation_name, path) is None:
        demons = SItk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(steps)
        demons.SetStandardDeviations(sigma)

        if verbose > 1:
            demons.AddCommand(
                SItk.sitkIterationEvent,
                lambda: print('\033[K> Demons %d: %f' % (
                    demons.GetElapsedIterations(), demons.GetMetric()
                ))
            )

        deformation_field = demons.Execute(fixed_float32, moving_float32)
        if verbose > 0:
            print('\033[KRegistration finished')

        if name is not None and path is not None:
            SItk.WriteImage(deformation_field, os.path.join(path, deformation_name))
    else:
        deformation_field = SItk.ReadImage(os.path.join(path, deformation_name))

    return SItk.GetArrayFromImage(deformation_field)
