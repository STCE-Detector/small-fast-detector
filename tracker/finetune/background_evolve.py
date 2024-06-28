import os
import sys

os.system("nohup sh -c '" + sys.executable + f" evolve.py > hupba.txt 2>&1' &")
