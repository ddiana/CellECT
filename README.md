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





CellECT Segmentation Tool
=========================




CellECT Tracking Tool
=====================




CellECT Workspace Creation
==========================
