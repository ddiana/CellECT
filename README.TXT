CellECT: Cell Evolution Capturing Tool
======================================

For detailed documentation please visit: 
https://github.com/ddiana/CellECT#cellect-cell-evolution-capturing-tool


About
-----

CellECT is a tool for cell analysis in 3-D confocal microscopy membrane volumes. CellECT provides a segmentation tool, which runs seeded watershed on the volume, predicts uncertain areas and allows the user to interact with the segmentation in order to correct it. CellECT also provides a tracking tool for discovering cell lineage across multiple 3-D volumes.


CellECT Segmentation Tool
-------------------------

The CellECT Segmentation tool runs seeded Watershed segmentation, predicts areas of uncertainty and displays an "cellness" metric map of the segmentation. The user can select segments to correct, and provide feedback in another window. In the mean time, progress and status information is displayed in the terminal window. 

The user can interact with the segmentation by:
1) providing a seed to make a new segment
2) providing a seed to adjust an existing segment 
3) merging two existing segments 


CellECT Tracking Tool
---------------------

The tracking tool is work in progress and not published yet.


CellECT Workspace Creation Tool
-------------------------------

This tool generates the directories necessary for a workspace for a new project. Place your input files accordingly.

License and Disclaimer
----------------------

AUTHOR: 
Diana Delibaltov, http://ece.ucsb.edu/~diana/
Vision Research Lab, http://vision.ece.ucsb.edu/
University of California, Santa Barbara

LICENSE: UCSB, (c) 2013

DISCLAIMER: I assume no responsibility for any effect this software may have on you, your family, pet, computer, or anything else related to you or your existance. No warranty provided nor implied.



