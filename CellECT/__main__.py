import os
import sys

print "Calling seg_tool app."

os.system("python CellECT/seg_tool " + sys.argv[1])
