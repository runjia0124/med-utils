"""
This script does the following:
1. 'resize3d' resamples a 3d image to a give shape and voxel size
2. flip x-axis
3. organize and save bvec and bvals accordingly
"""
import numpy as np
import nibabel as nib
from scipy import ndimage
from dipy.io.image import save_nifti
import skimage.transform as skTrans
from dipy.io import read_bvals_bvecs
import scipy
from scipy.interpolate import RegularGridInterpolator, Rbf
from nilearn.image import concat_imgs, resample_img, mean_img

def resize3d(data, new_dim, method):

    if len(data.shape) != len(new_dim):
        raise Exception("Target dimension must be consistent with input data!")

    if len(data.shape) == 2:
        print('Do 2d resize...')
    elif len(data.shape) == 3:
        in_dim_x, in_dim_y, in_dim_z = data.shape
        out_dim_x, out_dim_y, out_dim_z = new_dim
        
        in_grid_x = np.linspace(0, 1, in_dim_x)
        in_grid_y = np.linspace(0, 1, in_dim_y)
        in_grid_z = np.linspace(0, 1, in_dim_z)
        xq, yq, zq = np.mgrid[0: 1+1/(out_dim_x-1)-1e-6: 1/(out_dim_x-1), 0: 1+1/(out_dim_y-1)-1e-6: 1/(out_dim_y-1), 0: 1+1/(out_dim_z-1)-1e-6: 1/(out_dim_z-1)]
        
        rgi = RegularGridInterpolator((in_grid_x, in_grid_y, in_grid_z), data, method=method)
        data_downsampled = rgi((xq, yq, zq))
        data_downsampled = np.int16(data_downsampled)

    return data_downsampled

if __name__ == '__main__':

    subj_name = '103414'
    filename = '/Users/runjia/Code/py-sttar/aparc+aseg.nii.gz'
    # bvec_name = '/home/runjia/Code/sttar_code/Data_HCPS1200/2mm_ACPC/103414_old_ver/raw/bvecs'
    # bval_name = '/home/runjia/Code/sttar_code/Data_HCPS1200/2mm_ACPC/103414_old_ver/raw/bvals'
    path_fdt = ''
    # for aseg
    dim_orig = [260, 311, 260]
    res_orig = [0.7, 0.7, 0.7]
    # for brain mask
    # dim_orig = [145, 174, 145]
    # res_orig = [1.25, 1.25, 1.25]
    res_new = [2, 2, 2]

    img = nib.load(filename)
    data = img.get_fdata()
    # affine = img.affine
    xform = np.eye(4) * 2
    for i in range(0, 3):
        xform[i][3] = 2
    xform[3][3] = 1
    affine = xform
    dim_new = [a*b/c for a, b, c in zip(dim_orig, res_orig, res_new)]
    dim_new = [round(i) for i in dim_new]

    # T1w/DWI data type should be INT16 to reduce data size, mask unsigned 8 bit, aseg signed 16 bit
    # new_data = np.zeros((dim_new[0], dim_new[1], dim_new[2], data.shape[3]), dtype=np.int32)
    new_data = np.zeros((dim_new[0], dim_new[1], dim_new[2], 1), dtype=np.int16)
    # new_data = np.zeros((dim_new[0], dim_new[1], dim_new[2], 1), dtype=np.uint8)
    print('Original data shape: ', data.shape[:3])
    print('New dimension: ', dim_new)
    print('Processing slice...')
    # from Ziyang
    # tsuika comments: code from Ziyang works, also try https://github.com/sacmehta/3D-ESPNet/blob/master/Transforms.py, and scipy.interpolate
    # for resampling brain mask, multiply by 255 to increase voxel intensity
    # data = data * 255
    # new_data[..., 0] = ndimage.zoom(data, (dim_new[0]/dim_orig[0], dim_new[1]/dim_orig[1], dim_new[2]/dim_orig[2]),
    #                                 order=1, mode='constant')
    new_data[..., 0] = resize3d(data, dim_new, method='nearest')
    new_data = np.squeeze(new_data)
    # for mask
    # new_data[new_data > 0] = 1

    # flip on the first axis?
    # new_data = np.flip(new_data, 0)
    # save_nifti('./nodif_brain_mask.nii', new_data, affine)
    save_nifti('./raparc+aseg_myself.nii', new_data, affine)

    """
    raw_bval, raw_bvec = read_bvals_bvecs(bval_name, bvec_name)
    print(raw_bvec.shape)

    # flip x-axis for bvec
    bvec_flip_x = raw_bvec
    bvec_flip_x[..., 0] = -bvec_flip_x[..., 0]

    # save b_vec
    with open('./bvecs_flip_x', 'a') as f:
        for dim in range(3):
            for vec in range(bvec_flip_x.shape[0]):
                if vec == bvec_flip_x.shape[0] - 1:
                    f.write("{:.6f}\n".format(bvec_flip_x[vec][dim]))
                else:
                    f.write("{:.6f}\t".format(bvec_flip_x[vec][dim]))

    # do nothing for bval
    with open('./bvals', 'a') as f:
        for idx in range(len(raw_bval)):
            if idx == len(raw_bval) - 1:
                f.write(f"{int(raw_bval[idx])}")
            else:
                f.write(f"{int(raw_bval[idx])} ")
    """







