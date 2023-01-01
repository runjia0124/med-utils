"""This whole script is adapted from QSIPrep.

Performs the following:
1. reorient an image to LPS (optional)
2. resample given target zooms (voxel size), target shape

"""
import numpy as np
import os.path as op
import nibabel as nib
from nipype import logging
import nilearn.image as nli
from bids import DerivativesDataSink
from dipy.io.image import save_nifti

LOGGER = logging.getLogger('nipype.interface')


def to_lps(input_img, new_axcodes=("L", "P", "S")):
    if isinstance(input_img, str):
        input_img = nib.load(input_img)
    input_axcodes = nib.aff2axcodes(input_img.affine)
    # Is the input image oriented how we want?
    if not input_axcodes == new_axcodes:
        # Re-orient
        input_orientation = nib.orientations.axcodes2ornt(input_axcodes)
        desired_orientation = nib.orientations.axcodes2ornt(new_axcodes)
        transform_orientation = nib.orientations.ornt_transform(input_orientation,
                                                               desired_orientation)
        reoriented_img = input_img.as_reoriented(transform_orientation)
        return reoriented_img
    else:
        return input_img


def fname_presuffix(fname, prefix="", suffix="", newpath=None, use_ext=True):
    """Manipulates path and name of input filename

    Parameters
    ----------
    fname : string
        A filename (may or may not include path)
    prefix : string
        Characters to prepend to the filename
    suffix : string
        Characters to append to the filename
    newpath : string
        Path to replace the path of the input fname
    use_ext : boolean
        If True (default), appends the extension of the original file
        to the output name.

    Returns
    -------
    Absolute path of the modified filename

    >>> from nipype.utils.filemanip import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'

    >>> from nipype.interfaces.base import Undefined
    >>> fname_presuffix(fname, 'pre', 'post', Undefined) == \
            fname_presuffix(fname, 'pre', 'post')
    True

    """
    pth, fname, ext = split_filename(fname)
    if not use_ext:
        ext = ""

    # No need for isdefined: bool(Undefined) evaluates to False
    if newpath:
        pth = op.abspath(newpath)
    return op.join(pth, prefix + fname + suffix + ext)


def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : str
        file or path name

    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extension from fname

    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'

    """

    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]

    pth = op.dirname(fname)
    fname = op.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)

    return pth, fname, ext


def conform(fname, target_zooms, target_shape, deoblique_header, interpolation='continuous'):
    """Conform a series of T1w images to enable merging.

    Performs two basic functions:

    1. Orient to LPS (right-left, anterior-posterior, inferior-superior), skipped in this function
    2. Resample to target zooms (voxel sizes) and shape (number of voxels)

    target_zooms: target voxel size
    interpolation: 'continuous' for DWI/T1w, 'nearest' for aparc+aseg
    """

    # Load image, orient as LPS
    orig_img = nib.load(fname)
    # reoriented = to_lps(orig_img)
    reoriented = orig_img

    # Set target shape information
    target_span = np.array(target_shape) * np.array(target_zooms)

    zooms = np.array(reoriented.header.get_zooms()[:3])
    shape = np.array(reoriented.shape[:3])

    # Reconstruct transform from orig to reoriented image
    ornt_xfm = nib.orientations.inv_ornt_aff(
        nib.io_orientation(reoriented.affine), orig_img.shape)
    # Identity unless proven otherwise
    target_affine = reoriented.affine.copy()
    conform_xfm = np.eye(4)
    # conform_xfm = np.diag([-1, -1, 1, 1])

    xyz_unit = reoriented.header.get_xyzt_units()[0]
    if xyz_unit == 'unknown':
        # Common assumption; if we're wrong, unlikely to be the only thing that breaks
        xyz_unit = 'mm'

    # Set a 0.05mm threshold to performing rescaling
    atol = {'meter': 1e-5, 'mm': 0.01, 'micron': 10}[xyz_unit]

    # Rescale => change zooms
    # Resize => update image dimensions
    rescale = not np.allclose(zooms, target_zooms, atol=atol)
    resize = not np.all(shape == target_shape)
    if rescale or resize:
        if rescale:
            scale_factor = target_zooms / zooms
            target_affine[:3, :3] = reoriented.affine[:3, :3].dot(np.diag(scale_factor))

        if resize:
            # The shift is applied after scaling.
            # Use a proportional shift to maintain relative position in dataset
            size_factor = target_span / (zooms * shape)
            # Use integer shifts to avoid unnecessary interpolation
            offset = (reoriented.affine[:3, 3] * size_factor - reoriented.affine[:3, 3])
            target_affine[:3, 3] = reoriented.affine[:3, 3] + offset.astype(int)

        data = nli.resample_img(reoriented, target_affine, target_shape, interpolation=interpolation).get_data()
        conform_xfm = np.linalg.inv(reoriented.affine).dot(target_affine)
        reoriented = reoriented.__class__(data, target_affine, reoriented.header)

    if deoblique_header:
        is_oblique = np.any(np.abs(nib.affines.obliquity(reoriented.affine)) > 0)
        if is_oblique:
            LOGGER.warning("Removing obliquity from image affine")
            new_affine = reoriented.affine.copy()
            new_affine[:, :-1] = 0
            new_affine[(0, 1, 2), (0, 1, 2)] = reoriented.header.get_zooms()[:3] \
                * np.sign(reoriented.affine[(0, 1, 2), (0, 1, 2)])
            reoriented = nib.Nifti1Image(reoriented.get_fdata(), new_affine, reoriented.header)

    # Image may be reoriented, rescaled, and/or resized
    if reoriented is not orig_img:
        out_name = fname_presuffix(fname, suffix='_lps', newpath=None)
        reoriented.to_filename(out_name)
        print('Output file name: ', out_name)
        transform = ornt_xfm.dot(conform_xfm)
        if not np.allclose(orig_img.affine.dot(transform), target_affine):
            LOGGER.warning("Check alignment of anatomical image.")

    else:
        out_name = fname
        transform = np.eye(4)

    # mat_name = fname_presuffix(fname, suffix='.mat', newpath=runtime.cwd, use_ext=False)
    # np.savetxt(mat_name, transform, fmt='%.08f')

    return reoriented


if __name__ == '__main__':

    """
    An example to resample 'aparc+aseg.nii.gz' of voxel size 0.7mm, shape (260, 311, 260) to
    voxel size 2, shape (compute accordingly) 
    """
    filename = '/Users/runjia/Code/py-sttar/aparc+aseg.nii.gz'
    # img = nib.load(filename)
    # # reorient to LPS
    # img = to_lps(img)
    # data = img.get_fdata().astype(np.int32)
    # affine = img.affine
    # save_nifti('./test_resample_voxel.nii.gz', data, affine)
    reoriented = conform(fname=filename, target_zooms=(2, 2, 2), target_shape=(int(260*0.7/2), int(311*0.7/2), int(260*0.7/2)), deoblique_header=True)
