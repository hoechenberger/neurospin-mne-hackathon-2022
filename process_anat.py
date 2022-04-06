import mne
from mne_bids import BIDSPath, write_anat, get_anat_landmarks

subject = "sub-09"
subjects_dir = "./bids_data/derivatives/freesurfer/subjects"

# mne.bem.make_watershed_bem(subject, subjects_dir)
# mne.bem.make_scalp_surfaces(subject, subjects_dir)
# mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
# mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)


# t1w_bids_path = BIDSPath(subject="09", root="bids_data", extension="nii.gz",
#                          suffix='T1w')

# t1_fname = "./pogs_meg/ll_180197/220217/T1.nii.gz"
# info = mne.io.read_info('./pogs_meg/ll_180197/220217/run1_raw.fif')
# trans = mne.read_trans('trans/sub-09-trans.fif')

# # use ``trans`` to transform landmarks from the ``raw`` file to
# # the voxel space of the image
# landmarks = get_anat_landmarks(
#     t1_fname,  # path to the MRI scan
#     info=info,  # the MEG data file info from the same subject as the MRI
#     trans=trans,  # our transformation matrix
#     fs_subject=subject,  # FreeSurfer subject
#     fs_subjects_dir=subjects_dir,  # FreeSurfer subjects directory
# )

# # We use the write_anat function
# t1w_bids_path = write_anat(
#     image=t1_fname,  # path to the MRI scan
#     bids_path=t1w_bids_path,
#     landmarks=landmarks,  # the landmarks in MRI voxel space
#     verbose=True  # this will print out the sidecar file
# )
