#!/usr/bin/env python

import os, sys
from distutils.core import setup
from CellECT.seg_tool import module_info

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
                    'CellECT/seg_tool/seg_utils' ]

LONG_DESC = """
CellECT: Cell Evolution Capturing Tool.
"""

setup(name='CellECT',\
	version = module_info.__version__ ,\
	description = 'CellECT: Cell Evolution Capturing Tool', \
	long_description = LONG_DESC,\
	author = 'Diana Delibaltov',\
	license = 'UCSB',\
	author_email = 'diana.delibaltov@gmail.com',\
	url = 'http://vision.ece.ucsb.edu',\
	packages = CellECT_packages,\
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