import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
#print(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)