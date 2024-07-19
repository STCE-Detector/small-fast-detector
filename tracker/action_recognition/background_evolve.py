import os
import sys

os.system("nohup sh -c '" + sys.executable + f" evolve.py > general.txt 2>&1' &")
