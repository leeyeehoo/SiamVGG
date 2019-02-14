#!/usr/bin/python
#import torch

import vot
import sys
import time
from src.siamvggtracker import SiamVGGTracker
# *****************************************
# VOT: Create VOT handle at the beginning
#      Then get the initializaton region
#      and the first image
# *****************************************


handle = vot.VOT("rectangle")
selection = handle.region()

# Process the first frame
imagefile = handle.frame()
tracker = SiamVGGTracker(imagefile,selection)

if not imagefile:
    sys.exit(0)
    
while True:
    # *****************************************
    # VOT: Call frame method to get path of the 
    #      current image frame. If the result is
    #      null, the sequence is over.
    # *****************************************
    imagefile = handle.frame()
    if not imagefile:
        break
    region,confidence = tracker.track(imagefile)
    region = vot.Rectangle(region.x,region.y,region.width,region.height)
    # *****************************************
    # VOT: Report the position of the object 
    #      every frame using report method.
    # *****************************************
    handle.report(region,confidence)

