import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours
import os

import pylidc as pl
from pylidc.utils import consensus

# Query for all CT scans
scans = pl.query(pl.Scan)
print("total scans in Database: " + str(scans.count()))

for subjectID in range(848,scans.count()): #this should be 1, scans.count()
    s = 'LIDC-IDRI-%04i' % subjectID
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == s).first()
    if scans:
        vol = scans.to_volume()
        nods = scans.cluster_annotations()
        print(s + " has: " + str(len(scans.annotations)) + " annotations. ")

        nods = scans.cluster_annotations()
        for i,nod in enumerate(nods):
            i+1

            savefile = str(pathlib.Path().absolute()) + r"\database"

            if (nods[i][int(len(nods[i])/2)].malignancy == 5):
                print("cancerous")
                savefile = savefile + r"\cancerous"

            if (nods[i][int(len(nods[i])/2)].malignancy == 4):
                print("likely cancerous")
                savefile = savefile + r"\likely cancerous"

            if (nods[i][int(len(nods[i])/2)].malignancy == 3):
                print("probably cancerous")
                savefile = savefile + r"\probably cancerous"

            if (nods[i][int(len(nods[i])/2)].malignancy == 2):
                print("unlikely cancerous")
                savefile = savefile + r"\unlikely cancerous"
            
            if (nods[i][int(len(nods[i])/2)].malignancy == 1):
                print("safe")
                savefile = savefile + r"\safe"

            if os.path.isdir(savefile) == False:
                os.makedirs(savefile)
            
            print(savefile)

            cmask,cbbox,masks = consensus(nods[i], clevel=0.5, pad=[(20,20), (20,20), (0,0)])
            k = int(0.5*(cbbox[2].stop - cbbox[2].start))
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            ax.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=1.0)
        
            # Plot the annotation contours for the kth slice.
            colors = ['r', 'g', 'b', 'y']
            for j in range(len(masks)):
                for c in find_contours(masks[j][:,:,k].astype(float), 0.5):
                    label = "Annotation %d" % (j+1)
                    #plt.plot(c[:,1], c[:,0], colors[j], label=label)

            ax.axis('off')
            #ax.legend()
            plt.tight_layout()
            plt.savefig(savefile + "\\" + str(s) + "-" + str(i), bbox_inches="tight")
            #plt.show()
            plt.close()
