CellECT: Cell Evolution Capturing Tool
======================================

About
-----

CellECT is a tool for cell analysis in 3-D confocal microscopy membrane volumes. CellECT provides a segmentation tool, which runs seeded watershed on the volume, predicts uncertain areas and allows the user to interact with the segmentation in order to correct it. CellECT also provides an analysis tool which quantifies pattens over cells in a time series.

![CellECT: Cell Evolutio Capturing Tool](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/CellECT.png "CellECT: Cell Evolution Capturing Tool")


Installation
------------

CellECT is supported for Linux and MacOS, and was tested on Ubuntu 11.10, Ubuntu 12.04 and Fedora 18.

Prerequisites:

* python 2.7
* numpy 1.6.1
* scipy 0.10.0
* python-image
* matplotlib 1.1.1 or above
* pyml 0.7.10 (and libsvm)
* termcolor
* pylibtiff (and libtiff)
* pyside wrapper for Qt
* opencv python wrappers
* matlab (needed for imimposemin, watershed)
* scikit-learn
* pyside
* python-graph

An install script tested on Fedora 18 is provided [here](https://github.com/ddiana/CellECT/blob/master/prereqs_install_fedora.sh)

After the prerequisites, CellECT can be installed as follows:

```
python setup.py build
python setup.py install
```

Running CellECT
----------------

CellECT can be run from command line:

```
CellECT
```

CellECT can work with TIFF stacks which contain membrane channel (and optionally nuclei channel), at one or multiple time points. You can create a workspace from a new TIFF dataset, or you can open an existing workspace and continue working with it. 


![CellECT: New workspace](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/new_ws.png "CellECT: New workspace")

A new workspace will optionally take as input the output of a nuclear detector as a .csv file. If no such file is provided, the user can manually add points later, or or opt for randomized input points. Segmentations can be created from input seed points and/or from user markers. Existing segmentations can be adjusted with user interaction and propagated to neighboring time points. Time series analysis of patterns in the cells can be done using the analysis tool. Sets of neighboring time points can be batched processed in non-interactive mode and propagation mode.


CellECT: Segmentation Tool
==========================

The segmentation tool allows users to computer segmentations for 3D volumes offline or interactively. Segmentations may be corrected with user interaction. A "cellness metric" algorithm highlights problematic cells and learns from user feedback. Suggestions for merging pairs of neighboring cells are available. Once a satisfactory segmentation is obtained it can be propagated to neighoring time stamps and further edited if needed. 

How to Use the CellECT Segmentation Tool
----------------------------------------

The CellECT Segmentation tool runs seeded Watershed segmentation, predicts areas of uncertainty and displays an "cellness" metric map of the segmentation. The user can select segments to correct, and provide feedback in another window. In the mean time, progress and status information is displayed in the terminal window. A typical run of this application is shown in the figure below.

![CellECT Segmentation Tool](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/suggestion.png "CellECT Segmentation Tool")


The user interface window from the figure below is displayed once the segmentation tool is set up and ready to receive user input. The first panel shows a slice through the original volume. Panel 2 shows a slice through the segmentation, color coded by confidence in the segmentation (“cellness metric”): the segments colored in green are likely to be correct, and the segments colored in yellow-red are likely to be incorrect. The third panel shows the segmentation label map color coded by segment label. Finally, the last panel shows the difference between the current segmentation and the previous one (if any). Edges which were removed are colored in red. Edges which were added are colored in green.

![Main Interactive Segmentation Interface](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/uncertainty.png "Main Interactive Segmentation Interface")

The user can load the latest segmentation (if any), save the current segmentation, or make changes to the current segmentation. To make changes to the current segmentation one can click on any of the segments in panels 2 and 3. The z-slider at the bottom can be used to visualize the segmentation at other slices in the volume. Note that the segments that are marked as low confidence may appear correct, however the error might be present in a slice through the volume which is not visible in the current display.

Once a segment is clicked for correction, the window in the figure below shows a cropped region around the segment of interest. Three actions can be performed:

1.  Add seeds for a new label. Left click for one seed for one new label. The new seed will be marked by a star symbol.
2.  Add seeds for an old label. Right click for the label of interest. Left click to place a few seeds. The new seeds will be marked with star symbols.
3.  Merge two labels. Place two right clicks, one for each label to me merged.
4.  Mark a segment for delition.

Once the user has given enough corrections for this iteration, the "Rerun with user feedback" button in the main interactive segmentation window needs to be pressed for the next iteration to take place. This process repeats until the segmentation is satisfactory.


Information in the Console:
---------------------------

Refer to the terminal window for status information. The application prints useful information to the terminal window.

**Example 1:** The progress (and duration) of each task is displayed. For slow tasks the percentage of execution is displayed and updated.

![Progress and duration information is displayed in the terminal](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/seg_tool1.png "Progress and duration information is displayed in the terminal")


**Example 2:** Terminal shows user click information. This can be useful to check results from previous clicks and to make sure the input was correct.

![User clicks information is displayed in the terminal](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/seg_tool_clicks.png "User clicks information is displayed in the terminal")


**Example 3:** The terminal shows when MATLAB is called to run watershed.

![Matlab system call progress information is displayed in the terminal](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/seg_tool_matlab.png "Matlab system call progress information is displayed in the terminal")


**Example 4:** If the user chooses to save or load a segmentation, this progress is displayed in the terminal window. The user is also prompted for a final save before the application exits.

![Save and load information is displayed in the terminal](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/seg_tool_saving.png "Save and load information is displayed in the terminal")

Warnings regarding the user feedback, such as "Bad or no label" or "No file to load" are displayed in the terminal.



CellECT: Analysis Tool
==========================

The analysis tool can be launched from the main interface by selecting a subset of time points in the left panel and launching the "Analysis" tool. This displays trends and patterns of measurements computed in the cells of the existing segmentations.


![CellECT: Analysis tool](https://raw.github.com/ddiana/CellECT/master/CellECT/doc/md_figures/gui_analysis.png "CellECT: Analysis tool")


Disclaimer
======================

**Author:**

[Diana Delibaltov](http://ece.ucsb.edu/~diana) developed this software as a Ph.D. student at the [Vision Research Lab](http://vision.ece.ucsb.edu) at University of California, Santa Barbara.


**Disclaimer:**

I assume no responsibility for any effect this software may have on you,
your family, pet, computer, or anything else related to you or your existance.
No warranty provided nor implied.

