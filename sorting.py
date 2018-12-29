'''
You can get the data via:
wget http://pjreddie.com/media/files/cifar.tgz    
Important: Before proceeding, the student must reorganize the downloaded dataset files to match the expected directory structure, so that there is a dedicated folder for each class under 'test' and 'train', e.g.:
* test/airplane/airplane-1001.png
* test/bird/bird-1043.png

* train/bird/bird-10018.png
* train/automobile/automobile-10000.png
'''

from pathlib import Path
from typing import Tuple#List#, Set, Dict, Tuple, Optional

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'automobile')

def sort_files(sourceDir: str, classes: Tuple) -> None:
    PATH = Path(sourceDir)
    
    for name in classes:
        clsdir = PATH/name
        clsdir.mkdir(exist_ok=True)
        pattern = r'*'+name+'.png'
        for file in PATH.glob(pattern):
            newfile = clsdir/(file.parts[-1])
            file.replace(newfile)

#test
# sort_files('temp', ('frog','dog'))

#main run
sort_files('test', classes)
sort_files('train', classes)