"""
Practical functions for the STTAR project
"""
import os
import numpy as np
import nibabel as nib
from nibabel.streamlines.header import Field
from nibabel.streamlines.tractogram import Tractogram
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.orientations import (aff2axcodes, axcodes2ornt)


template_header = {'magic_number': b'TRACK',
		  'dimensions': np.array([91, 109, 91], dtype=np.int16),
		  'voxel_sizes': np.array([2., 2., 2.], dtype=np.float32),
		  'origin': np.array([0., 0., 0.], dtype=np.float32),
		  'nb_scalars_per_point': 0,
		  'scalar_name': np.array([b'', b'', b'', b'', b'', b'', b'', b'', b'', b''], dtype='|S20'),
		  'nb_properties_per_streamline': 0,
		  'property_name': np.array([b'', b'', b'', b'', b'', b'', b'', b'', b'', b''], dtype='|S20'),
		  'voxel_to_rasmm': np.array([[2., 0., 0., 0.],
									  [0., 2., 0., 0.],
									  [0., 0., 2., 0.],
									  [0., 0., 0., 1.]], dtype=np.float32),
		  'reserved': b'', 'voxel_order': b'LPS',
		  'pad2': b'LPS', 'image_orientation_patient': np.array([1., 0., 0., 0., 1., 0.], dtype=np.float32),
		  'pad1': b'', 'invert_x': b'', 'invert_y': b'', 'invert_z': b'', 'swap_xy': b'', 'swap_yz': b'',
		  'swap_zx': b'', 'nb_streamlines': 10, 'version': 2, 'hdr_size': 1000, 'endianness': '<', '_offset_data': 1000}


def compare_array(arr_1, arr_2):
	dim = arr_1.shape
	for i in range(dim[0]):
		for j in range(dim[1]):
			for k in range(dim[2]):
				if arr_1[i][j][k] != arr_2[i][j][k]:
					return False
	return True


def lps_ras(input_axcodes=("L", "P", "S"), new_axcodes=("R", "A", "S"), target_dim=None):

	if target_dim is None:
		raise ValueError("Target dimension is not specified")

	target_affine = np.eye(4)
	# Is the input image oriented how we want?
	if not input_axcodes == new_axcodes:
		# Re-orient
		input_orientation = nib.orientations.axcodes2ornt(input_axcodes)
		desired_orientation = nib.orientations.axcodes2ornt(new_axcodes)
		transform_orientation = nib.orientations.ornt_transform(desired_orientation,
																input_orientation)
		target_affine = nib.orientations.inv_ornt_aff(transform_orientation, target_dim)

	return target_affine


# nibabel.affines.apply_affine
def apply_affine(aff, pts):
	""" Apply affine matrix `aff` to points `pts`

	Returns result of application of `aff` to the *right* of `pts`.  The
	coordinate dimension of `pts` should be the last.

	For the 3D case, `aff` will be shape (4,4) and `pts` will have final axis
	length 3 - maybe it will just be N by 3. The return value is the
	transformed points, in this case::

		res = np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]
		transformed_pts = res.T

	This routine is more general than 3D, in that `aff` can have any shape
	(N,N), and `pts` can have any shape, as long as the last dimension is for
	the coordinates, and is therefore length N-1.

	Parameters
	----------
	aff : (N, N) array-like
		Homogenous affine, for 3D points, will be 4 by 4. Contrary to first
		appearance, the affine will be applied on the left of `pts`.
	pts : (..., N-1) array-like
		Points, where the last dimension contains the coordinates of each
		point.  For 3D, the last dimension will be length 3.

	Returns
	-------
	transformed_pts : (..., N-1) array
		transformed points

	Examples
	--------
	>>> aff = np.array([[0,2,0,10],[3,0,0,11],[0,0,4,12],[0,0,0,1]])
	>>> pts = np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
	>>> apply_affine(aff, pts) #doctest: +ELLIPSIS
	array([[14, 14, 24],
		   [16, 17, 28],
		   [20, 23, 36],
		   [24, 29, 44]]...)

	Just to show that in the simple 3D case, it is equivalent to:

	>>> (np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]).T #doctest: +ELLIPSIS
	array([[14, 14, 24],
		   [16, 17, 28],
		   [20, 23, 36],
		   [24, 29, 44]]...)

	But `pts` can be a more complicated shape:

	>>> pts = pts.reshape((2,2,3))
	>>> apply_affine(aff, pts) #doctest: +ELLIPSIS
	array([[[14, 14, 24],
			[16, 17, 28]],
	<BLANKLINE>
		   [[20, 23, 36],
			[24, 29, 44]]]...)
	"""
	aff = np.asarray(aff)
	pts = np.asarray(pts)
	shape = pts.shape
	pts = pts.reshape((-1, shape[-1]))
	# rzs == rotations, zooms, shears
	rzs = aff[:-1, :-1]
	trans = aff[:-1, -1]
	res = np.dot(pts, rzs.T) + trans[None, :]
	return res.reshape(shape)


def get_affine_rasmm_to_trackvis(header):
	return np.linalg.inv(get_affine_trackvis_to_rasmm(header))


def get_affine_trackvis_to_rasmm(header):
	""" Get affine mapping trackvis voxelmm space to RAS+ mm space

	The streamlines in a trackvis file are in 'voxelmm' space, where the
	coordinates refer to the corner of the voxel.

	Compute the affine matrix that will bring them back to RAS+ mm space, where
	the coordinates refer to the center of the voxel.

	Parameters
	----------
	header : dict or ndarray
		Dict or numpy structured array containing trackvis header.

	Returns
	-------
	aff_tv2ras : shape (4, 4) array
		Affine array mapping coordinates in 'voxelmm' space to RAS+ mm space.
	"""
	# TRK's streamlines are in 'voxelmm' space, we will compute the
	# affine matrix that will bring them back to RAS+ and mm space.
	affine = np.eye(4)

	# The affine matrix found in the TRK header requires the points to
	# be in the voxel space.
	# voxelmm -> voxel
	scale = np.eye(4)
	scale[range(3), range(3)] /= header[Field.VOXEL_SIZES]
	affine = np.dot(scale, affine)

	# TrackVis considers coordinate (0,0,0) to be the corner of the
	# voxel whereas streamlines returned assumes (0,0,0) to be the
	# center of the voxel. Thus, streamlines are shifted by half a voxel.
	offset = np.eye(4)
	offset[:-1, -1] -= 0.5
	affine = np.dot(offset, affine)  # coord / 2 - 0.5, -> voxel space

	# If the voxel order implied by the affine does not match the voxel
	# order in the TRK header, change the orientation.
	# Runjia comments: actually bring streamlines to header space (LPS voxel here),
	# so can be transformed to ras+mm by multiplying the header.to_ras_mm
	# TODO: Does this mean the header.to_ras_mm should be a LPS2RAS_voxel2mm matrix? update: streamlines are in RPI space
	vox_order = header[Field.VOXEL_ORDER]
	# Input header can be dict or structured array
	if hasattr(vox_order, 'item'):  # structured array
		vox_order = header[Field.VOXEL_ORDER].item()
	affine_ornt = "".join(aff2axcodes(header[Field.VOXEL_TO_RASMM]))
	header_ornt = axcodes2ornt(vox_order.decode('latin1').upper())
	affine_ornt = axcodes2ornt(affine_ornt)
	ornt = nib.orientations.ornt_transform(header_ornt, affine_ornt)
	M = nib.orientations.inv_ornt_aff(ornt, header[Field.DIMENSIONS])
	affine = np.dot(M, affine)

	# Applied the affine found in the TRK header.
	# voxel -> rasmm
	voxel_to_rasmm = header[Field.VOXEL_TO_RASMM]
	affine_voxmm_to_rasmm = np.dot(voxel_to_rasmm, affine)
	return affine_voxmm_to_rasmm.astype(np.float32)


def load_trk_with_streamline(filename):
	"""
	Load streamlines from .trk file,
	streamlines are in trackvis voxel space for the STTAR project

	:param filename: .trk file name
	:return: [nib-style tractogram obj, np.array-style streamlines]
	"""
	tractogram = nib.streamlines.load(filename, lazy_load=False)
	streamlines = tractogram.streamlines

	# transform streamlines from ras+mm(tgm.streamlines) to trackvis space
	affine_to_trackvis = get_affine_rasmm_to_trackvis(tractogram.header)
	for i in range(len(streamlines)):
		streamlines[i] = apply_affine(affine_to_trackvis, streamlines[i])

	return tractogram, streamlines


def save_streamline(save_fn, streamlines, res, dim, header=None):
	"""
	Save streamlines in trackvis voxel space to .trk file,
	designed for the STTAR project, might be different from common practice

	:param save_fn: save file name
	:param streamlines: streamlines in trackvis voxel space
	:param res: resolution
	:param dim: dimension
	:param header: .trk header, usually no need to provide
	:return: .trk file
	"""

	# if header is not specified by user, use the designed template header
	if header is None:
		header = template_header

	fiber_bundle = ArraySequence(streamlines)
	save_tractogram = Tractogram(fiber_bundle, affine_to_rasmm=np.eye(4))

	header[Field.VOXEL_SIZES] = res
	header[Field.DIMENSIONS] = dim
	lps_rasmm = lps_ras(target_dim=header[Field.DIMENSIONS])
	lps_voxel_rasmm = lps_rasmm.copy()
	lps_voxel_rasmm[range(3), range(3)] *= res
	# TODO: added here, not sure if this is correct, this affine: RPI->LPS->RAS?
	header[Field.VOXEL_TO_RASMM] = lps_voxel_rasmm
	# set this transform is critical, because our streamlines are in mm space;
	# later on, nib.streamlines.save will implicitly do a to_world transformation using this matrix
	save_tractogram.affine_to_rasmm = get_affine_trackvis_to_rasmm(header)
	nib.streamlines.save(save_tractogram, save_fn,
						header=header)


def load_rpi(filename):
	"""Adapted from DPARSF-read_RPI.m,
	Conform label to RPI space, to align with streamlines

	:param filename: input file name
	:return: conformed data and img with ORIGINAL header (TODO: return updated header in the future)
	"""
	img = nib.load(filename)
	data = img.get_fdata()
	mat = img.affine[:3, :3]
	new_affine = img.affine.copy()

	# ony when affine does not involve rotation
	if sum(sum(mat - np.diag(np.diag(mat)))) == 0.:
		if mat[0][0] > 0: # L, need to convert to R
			data = np.flip(data, 0)
			new_affine[0, :] = -1 * img.affine[0, :]
		if mat[1][1] < 0: # A, need to convert to P
			data = np.flip(data, 1)
			new_affine[1, :] = -1 * img.affine[1, :]
		if mat[2][2] < 0: # S, need to convert to I
			data = np.flip(data, 2)
			new_affine[2, :] = -1 * img.affine[2, :]

	if sum(sum((new_affine - img.affine))) == 0.:
		print('No transformation performed')

	return data, img
