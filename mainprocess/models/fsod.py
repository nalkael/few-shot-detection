import os
import detectron2
import sys
import subprocess

import fsdet

if __name__ == '__main__':
    command = [
        'python3', '-m', 'demo.demo',
        '--config-file', 'configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml',
        '--input', 'demo/input1.jpg', 'demo/input2.jpg',
    ]
    
    subprocess.run(command)