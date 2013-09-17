CellECT: Segmentation Tool
==========================

CellECT Segmentation Tool is the interactive segmentation component of the CellECT tool for cell analysis in 3-D confocal microscopy membrane volumes. CellECT provides a segmentation tool, which runs seeded watershed on the volume, predicts uncertain areas and allows the user to interact with the segmentation in order to correct it. CellECT also provides a tracking tool for discovering cell lineage across multiple 3-D volumes. The segmentation tool allows the user to interract with the results of segmentation by adding, removing or modifying segments.

Back to [CellECT](https://github.com/ddiana/CellECT#cellect-cell-evolution-capturing-tool)

Move to [CellECT Tracking Tool](https://github.com/ddiana/CellECT/tree/master/CellECT/track_tool#cellect-segmentation-tool)


Installation
------------

See installation of [CellECT](https://github.com/ddiana/CellECT)


Using CellECT
-------------

CellECT can work with TIFF stacks which contain membrane channel (and optionally nuclei channel), at one or multiple time points. You can create a workspace from a new TIFF dataset, or you can open an existing workspace and continue working with it. A new workspace will optionally take as input the output of a nuclear detector as a .csv file. If no such file is provided, the user can manually add points later, or or opt for randomized input points.

The CellECT segmentation tool can be run on one individual time point at a time. Alternatively, the preliminary segmentation can be run on multiple time points, without any user interaction.


CellECT Segmentation Tool
=========================


How to Use the CellECT Segmentation Tool
----------------------------------------

The CellECT Segmentation tool runs seeded Watershed segmentation, predicts areas of uncertainty and displays an "cellness" metric map of the segmentation. The user can select segments to correct, and provide feedback in another window. In the mean time, progress and status information is displayed in the terminal window. A typical run of this application is shown in the figure below.

![CellECT Segmentation Tool](CellECT/doc/md_figures/full_screen.png "CellECT Segmentation Tool")


The user interface window from the figure below is displayed once the segmentation tool is set up and ready to receive user input. The first panel shows a slice through the original volume. Panel 2 shows a slice through the segmentation, color coded by confidence in the segmentation (“cellness metric”): the segments colored in green are likely to be correct, and the segments colored in purple are likely to be incorrect. The third panel shows the segmentation label map color coded by segment label. Finally, the last panel shows the difference between the current segmentation and the previous one (if any). Edges which were removed are colored in red. Edges which were added are colored in green.

![Main Interactive Segmentation Interface](CellECT/doc/md_figures/main_gui.png "Main Interactive Segmentation Interface")

The user can load the latest segmentation (if any), save the current segmentation, or make changes to the current segmentation. To make changes to the current segmentation one can click on any of the segments in panels 2 and 3. The z-slider at the bottom can be used to visualize the segmentation at other slices in the volume. Note that the segments that are marked as low confidence may appear correct, however the error might be present in a slice through the volume which is not visible in the current display.

Once a segment is clicked for correction, the window in the figure below shows a cropped region around the segment of interest. Three actions can be performed:

1.  Add seeds for a new label. Left click for one seed for one new label. The new seed will be marked by a star symbol.
2.  Add seeds for an old label. Right click for the label of interest. Left click to place a few seeds. The new seeds will be marked with star symbols.
3.  Merge two labels. Place two right clicks, one for each label to me merged.

![Segment Correction Interface](CellECT/doc/md_figures/correct_segment_gui.png "Correct Segment Interface")

Multiple such corrections can be made for each segment correction window. Multiple segment correction windows can be opened. Once the user has given enough corrections for this iteration, the main interactive segmentation window needs to be closed (along with any other remaining windows) for the next iteration to take place. This process repeats until the segmentation is satisfactory.



Information in the Console:
---------------------------

Refer to the terminal window for status information. The application prints useful information to the terminal window.

**Example 1:** The progress (and duration) of each task is displayed. For slow tasks the percentage of execution is displayed and updated.

![Progress and duration information is displayed in the terminal](CellECT/doc/md_figures/seg_tool1.png "Progress and duration information is displayed in the terminal")


**Example 2:** Terminal shows user click information. This can be useful to check results from previous clicks and to make sure the input was correct.

![User clicks information is displayed in the terminal](CellECT/doc/md_figures/seg_tool_clicks.png "User clicks information is displayed in the terminal")


**Example 3:** The terminal shows when MATLAB is called to run watershed.

![Matlab system call progress information is displayed in the terminal](CellECT/doc/md_figures/seg_tool_matlab.png "Matlab system call progress information is displayed in the terminal")


**Example 4:** If the user chooses to save or load a segmentation, this progress is displayed in the terminal window. The user is also prompted for a final save before the application exits.

![Save and load information is displayed in the terminal](CellECT/doc/md_figures/seg_tool_saving.png "Save and load information is displayed in the terminal")

Warnings regarding the user feedback, such as "Bad or no label" or "No file to load" are displayed in the terminal.




License and Disclaimer
======================

**Author:**

[Diana Delibaltov](http://ece.ucsb.edu/~diana), Ph.D. student at the [Vision Research Lab](http://vision.ece.ucsb.edu) at University of California, Santa Barbara.

**License:**


**Disclaimer:**

I assume no responsibility for any effect this software may have on you,
your family, pet, computer, or anything else related to you or your existance.
No warranty provided nor implied.

