CellECT: Cell Evolution Discovery Tool
======================================

About
-----
CellECT is a tool for cell analysis in 3-D confocal microscopy membrane volumes. CellECT provides a segmentation tool, which runs seeded watershed on the volume, predics uncertain areas and allows the user to interact with the segmentation in order to correct it. CellECT also provices a tracking tool for discovering cell lineage across multiple 3-D volumes.

Skip to [CellECT Segmentation Tool](https://github.com/ddiana/CellECT#cellect-segmentation-tool)

Skip to [CellECT Tracking Tool](https://github.com/ddiana/CellECT#cellect-tracking-tool)

Instalation
------------

CellECT is supported for Linux and MacOS, and was tested on Ubuntu 11.10, Ubuntu 12.04 and Fedora 18.

Prerequisittes:

* python 2.7
* numpy 1.6.1
* scipy 0.10.0
* python-image
* matplotlib 1.1.1 or above
* pyml 0.7.10 (and libsvm)
* termcolor
* matlab (needed for imimposemin, watershed)

After the prerequisttes, CellECT can be installed as follows:

```
python setup.py build
python setup.py install 
```

Running CellECT
----------------

CellECT can be run from command line:
```
CellECT [path-to-config-file]
```

The configuration file is optional and is passed to the segmentation tool if this option is selected. Otherwise, if not configuration file is provided and the segmentation tool option is selected, the user will be prompted to select the file through a dialog window.

When running CellECT the user can choose from the applications available:

* CellECT segmentation tool
* CellECT tracking tool
* CellECT workspace creation tool

![CellECT main menu](CellECT/doc/md_figures/cellect_menu.png "CellECT Main Menu")


Workspace Directory
-------------------

The CellECT tools operate in a predefined workspace directory structure. To create a directory structure for a new project, refer to the [workspace creation tool](https://github.com/ddiana/CellECT#cellect-workspace-creation).

The workspace directory must contain the following sub-directories:

* **config_files** (holds .cfg files where the information specific for this dataset is specified.
* **init_watershed_all_timespamps** (holds .mat files with the initial watershed segmentation and nuclei coordinates)
* **input_slices** (.png slices of the volume nucleus and membrane channels in the z-stack and t-stack. This is used by the tracking tool only)
* **segs_all_time_stamps** (stores the resulting segmentation)
* **tracker_config** (configuration files for the tracker tool only)
* **training_data** (stores .mat files containing example segmentations and the list of labels for positive and negative examples).

![Workspace structure](CellECT/doc/md_figures/workspace_directories.png "Workspace directories")




CellECT Segmentation Tool
=========================




Input Data
----------

The following information needs to present in one (or several) .mat files. The mat file containing these items and the variable names that refer to them are specified in the configuration file. All path information should be relative to the workspace root directory.

* _volume_mat_path_ = path to mat file containing 3-D array holding the membrane channel of the volume to segment
* _volume_mat_var_ = name of variable containing the holding the membrane channel of the volume to segment
* _first_seg_mat_path_ = path to mat file containing 3-D array of the same size containing the label map of the initial watershed segmentation
* _first_seg_mat_var_ = name of variable containing 3-D array of the same size containing the label map of the initial watershed segmentation
* _nuclei_mat_path_ = path to mat file containing 2-D array of x-y-z coordinates corresponding to the nuclei.
* _nuclei_mat_var_ =  name of variable containing 2-D array of x-y-z coordinates corresponding to the nuclei.
* _training_vol_mat_path_ = path to mat file containing the 3-D array holding the membrane channel of the volume used for training examples.
* _training_vol_mat_var_ = name of variable containing the 3-D array holding the membrane channel of the volume used for training examples.
* _training_vol_nuclei_mat_path_ = path to mat file containing 2-D array of the x-y-z coordinates of the nuclei used in the training segmentation
* _training_vol_nuclei_mat_var_ = name of variable containing 2-D array of the x-y-z coordinates of the nuclei used in the training segmentation
* _training_positive_seg_mat_path_ = path to mat file containing 3-D array holding the label map used for positive examples.
* _training_positive_seg_mat_var_ = name of variable containing 3-D array holding the label map used for positive examples.
* _training_positive_labels_mat_path_ = path to mat file containing 1-D array listing all the labels of segments used for positive examples
* _training_negative_seg_mat_path_ = path to mat file containing 3-D array holding the label map used for negative examples.
* _training_negative_seg_mat_var_ = name of variable containing 3-D array holding the label map used for negative examples.
* _training_negative_labels_mat_path_ = path to mat file containing 1-D array listing all the labels of segments used for negative examples
* _save_location_prefix_ = path to the segs_all_time_stamps directory in the workspace, and the prefix for this segmentation, where all the segmentation results will be stored. Example : old_ascidian_workspace/segs_all_time_stamps/timestamp_0_
* _has_bg_ = flag indicating if there is a background region in the volume (0 or 1)
* _use_size_ = flag indicating if the size of the segments should be used as a feature (0 or 1)
* _use_border_intensity_ = flag indicating if the border intensity should be used as a feature (0 or 1)
* _use_border_distance_ = flag indicating if the distance to the border should be used as a feature (0 or 1)

Store configuration file in **workspace/config_files**.

Store initial watershed segmentations in **workspace/init_watershed_all_time_stamps**.

Store training examples in **workspace/training_data**.

If you have to generate several such config files for multiple time stamps, refer to **CellECT/utils/prepare_config_files_for_timestamp.py** to automate this process.


Output Data
-----------

Output data is stored in **workspace/segs_all_time_stamps.**

Output files contain the following: (where prefix is defined by the user in the configuration file, e.g. timestamp_10)

* **prefix_label_map.mat** holds the segmentation label map.
* **prefix_nuclei.xml** holds the nuclei information and the relationships resulting from user interactions.
* **prefix_seeds.xml** holds the seed information and the relationships resulting from user interactions. This is used by the tracking tool.
* **prefix_seed_segment_props.xml** holds the watershed segment properties for those segments resulting from seeds. (prior to label reassignment) This is used by the tracking tool.
* **prefix_segment_props.xml** holds the final segment properties (after label reassignment)
* **prefix_z_%d_seg.png** label map slices used by the tracking tool.

Segmentation results can be saved and loaded from the segmentation tool application.



CellECT Tracking Tool
=====================

CellECT tracking tool for cell lineage is work in progress and not published yet.


CellECT Workspace Creation
==========================

The workspace creation tool is useful to create the directory skeleton structure that CellECT expects. You should use this when you want to apply CellECT to your own data. 

You can access this application from the CellECT menu. Alternatively, you can call this application directly as:

```
CellECT_create_workspace_directories [path_to_workspace, workspace_name]
```

If the path_to_workspace and workspace_name parameters are not provided, the application will request the user to celect a directory in which to create the workspace, as shown below. Next, the user will be prompted from the console to provide the desired workspace name.

![Workspace creation tool directory dialog](CellECT/doc/md_figures/workspace_creation_dialog.png "Workspace creation tool dialog")


License and Disclaimer
======================

UCSB license.

