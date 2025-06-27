# Welcome to LOMBA!

LOMBA is the french acronym for 'Locating Bacterial Flagellar Motors'. 

This is the repo for the group project of Le Wagon Data Science batch #1992. We will soon put the link to see the demo day presentation (in french) here.

## Description
We used the data for the [BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025). Here is the description of the challenge :

> The flagellar motor is a molecular machine that facilitates the motility of many microorganisms, playing a key role in processes ranging from chemotaxis to pathogenesis. Cryogenic electron tomography (cryo-ET) has enabled us to image these nanomachines in near-native conditions. But identifying flagellar motors in these three-dimensional reconstructions (tomograms) is labor intensive. Factors such as a low signal-to-noise ratio, variable motor orientations, and the complexity of crowded intracellular environments complicate automated identification. Cryo-ET studies become limited by the bottle-neck of a human in the loop. In this contest, your task is to develop an image processing algorithm that identifies the location of a flagellar motor, if it is present.

Tomograms are 3D reconstructed image from a set of 2D 'slices'. The challenge provided [this video](https://www.cellstructureatlas.org/6-2-flagellar-motor.html) to get a better understanding of tomograms within the scope of bacteria observations.

The evaluation metrics was a Fbeta score with beta = 2. Fbeta score is the weighted harmonic mean of precision and recall ; beta > 1 gives more weight to recall, while beta < 1 favors precision.
With beta = 2, the recall is twice as important as the precision. You can see below the formula for the Fbeta score, with tp as true positives and fn as false negatives:

![image](https://github.com/user-attachments/assets/7777e62c-e097-4e75-b59c-9945d09779f9)

Besides the presence of motor, the goal was to predict its position. If the prediction was too far away from the ground truth, the prediction will be considered as a false negative. The challenge set a threshold of 1000 Å (angstrom, 10e-10m).

## Our implementation
To cope with the limited time accorded to the project (~10 days), we reduced the scope of the challenge by:
  - considering only tomograms with 0 or 1 motor
  - to ease preprocessing, we only considered tomograms with shape X<960, Y<960 
  - we increased the prediction threshold from 1000 to 2000 Å

To answer the challenge we created 3 models:

![image](https://github.com/user-attachments/assets/da42082d-e96a-4359-b0c6-94d553cebf12)


The first two models used 2D images, obtained by creating a 'mean' image of the tomogram (mean pixel value accross all images of the tomogram) and use local equalieation thanks to the `exposure.equalize_adapthist` method from the scikit-image package to increase contrast.

We first predicted the presence of a motor in tomogram. We had a high fbeta score on this task: 0.90.

Then we predicted the X, Y position of the motor within the tomogram. We had a great decrease in our fbeta score, as it fell to : 0.64. There are two main reasons for a failed prediction of the position: either the distance between ground truth and prediction is greater than the threshold (11 % of tomograms classified as fn), or the regression model can't find the class corresponding to the motor (23 % of tomograms classified as fn). Indeed, we didn't directly predicted X and Y. We used the U-Net architecture which is used for semantic segmentation, typically in medical imaging or computer vision. It produces a segmentation map (pixel-by-pixel classification mask) from an input image. We deduced X, Y coordinates by averaging the coordinates of pixels corresponding to a given motor. For some tomograms the model was not able to attribute pixels to the motor class.

Then on the tomograms were we predicted X, Y we use a third model to which we gave 3D data to obtain the Z position, i.e. the slice where the motor was. We didn't loose much score at this step, as we ended with a fbeta score of: 0.62.

The path worth exploring would be to predict Z before X, Y ; indeed we may have a be better at this first task, and we could then try to predict X, Y on the predicted slice or a set of slice centered on the predicted slice.



## Contributors
<a href="https://github.com/mc-will/BYU_Locating-Bacterial_Flagellar_Motors_2025/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mc-will/BYU_Locating-Bacterial_Flagellar_Motors_2025" />
</a>
