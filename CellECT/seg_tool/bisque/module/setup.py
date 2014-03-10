# Install script for SeededWatershed3D
import sys
from bq.setup.module_setup import matlab_setup, read_config, python_setup
def setup(params, *args, **kw):
    python_setup('CellECT_seg_tool.py', params=params)  # CellECT main script
	python_setup('CellECT_SegTool.py',  params=params ) # python wrapper
    
if __name__ =="__main__":
    params = read_config('runtime-bisque.cfg')
    if len(sys.argv)>1:
        params = eval (sys.argv[1])
    sys.exit(setup(params))
