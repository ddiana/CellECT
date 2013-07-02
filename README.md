CellECT: Cell Evolution Discovery Tool
======================================

About
-----
CellECT is a tool for cell analysis in 3-D confocal microscopy membrane volumes. CellECT provides a segmentation tool, which runs seeded watershed on the volume, predics uncertain areas and allows the user to interact with the segmentation in order to correct it. CellECT also provices a tracking tool for discovering cell lineage across multiple 3-D volumes.

Instalation:
------------

CellECT is supported for Linux and MacOS, and was tested on Ubuntu 11.10, Ubuntu 12.04 and Fedora 18.

Prerequisittes:

* python 2.7 (modules: copy, time, sys, re, xml, pdb, os)
* numpy 1.6.1
* scipy 0.10.0
* python-image
* matplotlib 1.1.1 or above
* pyml 0.7.10 (and libsvm)
* termcolor
* matlab (imimposemin, watershed)

After installing the prerequisttes CellECT can be installed as follows:

```
python setup.py build
python setup.py install 
```

Running CellECT:
----------------

CellECT can be run from command line:
```
CellECT [path-to-config-file]
```

The configuration file 




CellECT Segmentation Tool
=========================




CellECT Tracking Tool
=====================

CellECT tracking tool for cell lineage is work in progress and not published yet.


CellECT Workspace Creation
==========================

The workspace creation tool is useful to create the directory skeleton structure that CellECT expects. You should use this when you want to apply CellECT to your own data. 

You can access this application from the CellECT menu. Alternatively, you can call this application directly as:
'''
CellECT_create_workspace_directories [path_to_workspace, workspace_name]
'''
If the path_to_workspace and workspace_name parameters are not provided, the application will request the user to celect a directory in which to create the workspace, as shown below. Next, the user will be prompted from the console to provide the desired workspace name.

![Workspace creation tool directory dialog](CellECT/doc/md_figures/workspace_creation_dialog.png "Workspace creation tool dialog")


License and Disclaimer
======================

UCSB license.

