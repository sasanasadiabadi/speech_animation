# speech-animation
implementing "A deep learning approach for generalized speech animation" paper using Julia and Knet...

refrence: https://dl.acm.org/citation.cfm?id=3073699

the project is implemented on GRID dataset available at http://spandh.dcs.shef.ac.uk/gridcorpus/

## Data collection
- Used OpenCV and Dlib to extract landmark points on lower face 
- Filter out position, scale and rotational effect (shape alignment)
    - General Procrustes Analysis
    
- shape alignment: ![](Results/alignment.png)

- Shape model
    - apply PCA on shape data 
   
- ![](Results/shapemodel.png)

- training data preperation (sliding widow) ![](Results/datasetprep.png)


