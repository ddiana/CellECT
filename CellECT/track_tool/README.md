CellECT: Tracking Tool
==========================

About
-----

CellECT Tracking Tool is the tracking component of the CellECT tool for cell analysis in 3-D confocal microscopy membrane volumes. CellECT provides a segmentation tool, which runs seeded watershed on the volume, predicts uncertain areas and allows the user to interact with the segmentation in order to correct it. CellECT also provides a tracking tool for discovering cell lineage across multiple 3-D volumes. The tracking tool constructs the cell lineage based on segmentations at every time point and provides multiple visualization tools.

Back to [CellECT](https://github.com/ddiana/CellECT#cellect-cell-evolution-capturing-tool)

Move to [CellECT Segmentation Tool](https://github.com/ddiana/CellECT/tree/master/CellECT/seg_tool#cellect-segmentation-tool)

Installation
------------

See installation of [CellECT](https://github.com/ddiana/CellECT)


Using CellECT
-------------

CellECT can work with TIFF stacks which contain membrane channel (and optionally nuclei channel), at one or multiple time points. You can create a workspace from a new TIFF dataset, or you can open an existing workspace and continue working with it. A new workspace will optionally take as input the output of a nuclear detector as a .csv file. If no such file is provided, the user can manually add points later, or or opt for randomized input points.

The CellECT tracking tool can be run on a subset of time points, selected from the list on the left of the UI. The tracking tool constructs the cell lineage and offers the multiple visualizations of the result, as illustrated in the screenshot below.

![CellECT Tracking Tool](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/CellECT_tracker.png "CellECT Tracking Tool")



License and Disclaimer
======================

**Author:**

[Diana Delibaltov](http://ece.ucsb.edu/~diana), Ph.D. student at the [Vision Research Lab](http://vision.ece.ucsb.edu) at University of California, Santa Barbara.


**License:**



**Disclaimer:**

I assume no responsibility for any effect this software may have on you,
your family, pet, computer, or anything else related to you or your existance.
No warranty provided nor implied.

