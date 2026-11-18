"""Keep only the prompted body span of the previously generated SAM masks."""
import argparse
from pathlib import Path
import numpy as np
from protocol import PROMPTS

p = argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('out',type=Path)
a = p.parse_args()
mask = np.load(a.source/'sam_mask.npy')
for i, (_,_,top,bottom) in enumerate(PROMPTS):
    mask[i,:top+3] = False
    mask[i,bottom-2:] = False
a.out.mkdir(parents=True)
np.save(a.out/'sam_mask.npy',mask)
