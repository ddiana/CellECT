#!/usr/bin/env python

import os, sys
from distutils.core import setup
from CellECT.seg_tool import module_info

print "running"

CellECT_packages = ['CellECT', 
                    'CellECT/seg_tool' ,
                    'CellECT/seg_tool/cellness_metric',
                    'CellECT/seg_tool/features',
                    'CellECT/seg_tool/gui', 
                    'CellECT/seg_tool/nuclei_collection',
                    'CellECT/seg_tool/run_watershed',
                    'CellECT/seg_tool/seed_collection',
                    'CellECT/seg_tool/seed_segment_collection',
                    'CellECT/seg_tool/seg_io',
                    'CellECT/seg_tool/segment_collection',
                    'CellECT/seg_tool/seg_utils',
                    'CellECT/seg_tool/bisque',
                    'CellECT/seg_tool/bisque/bisque_io',
                    'CellECT/track_tool/cell_tracker_core',
                    'CellECT/track_tool/gui',
                    'CellECT/track_tool/track_io',
                    'CellECT/track_tool/tissue_selection',
                    'CellECT/track_tool',
                    'CellECT/workspace_management',
                    'CellECT/gui' ]



LONG_DESC = """
CellECT: Cell Evolution Capturing Tool.
"""

setup(name='CellECT',\
    scripts = ['CellECT/CellECT', 'CellECT/seg_tool/CellECT_seg_tool', 'CellECT/track_tool/CellECT_track_tool', 'CellECT/utils/CellECT_create_workspace_directories'],\
	version = module_info.__version__ ,\
	description = 'CellECT: Cell Evolution Capturing Tool', \
	long_description = LONG_DESC,\
	author = 'Diana Delibaltov',\
	license = 'UCSB',\
	author_email = 'diana.delibaltov@gmail.com',\
	url = 'http://vision.ece.ucsb.edu',\
	packages = CellECT_packages,\
    package_data = {'CellECT': ['utils/*.m', 'utils/*.py' 'utils/fast_marching/*', 'README.TXT','track_tool/resources/gui_thumbnails/*', 'resources/*',  'data/training/ascidian/*', 'data/training/purdue/*']},\
	classifiers = [\
		'Development Status :: 3 - Alpha',\
		'Environment :: Console',\
		'Intended Audience :: Biologists',\
		'License :: UCSB',\
		'Operating System :: Linux, MacOS',\
		'Programming Language :: Python',\
		'Topic :: Cell Segmentation :: Tracking'\
		]
	)
