# Welcome to LOMBA!

LOMBA is the french acronym for 'Locating Bacterial Flagellar Motors'. 

This is the repo for the group project of Le Wagon Data Science batch #1992. We will soon put the link to see the demo day presentation (in french) here.

## Description
We used the data for the [BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025). Here is the description of the challenge :

> The flagellar motor is a molecular machine that facilitates the motility of many microorganisms, playing a key role in processes ranging from chemotaxis to pathogenesis. Cryogenic electron tomography (cryo-ET) has enabled us to image these nanomachines in near-native conditions. But identifying flagellar motors in these three-dimensional reconstructions (tomograms) is labor intensive. Factors such as a low signal-to-noise ratio, variable motor orientations, and the complexity of crowded intracellular environments complicate automated identification. Cryo-ET studies become limited by the bottle-neck of a human in the loop. In this contest, your task is to develop an image processing algorithm that identifies the location of a flagellar motor, if it is present.

Tomograms are 3D reconstructed image from a set of 2D 'slices'. The challenge provided [this video](https://www.cellstructureatlas.org/6-2-flagellar-motor.html) to get a better understanding of tomograms within the scope of bacteria observations.

The evaluation metrics was a Fbeta score with beta = 2. Fbeta score is the weighted harmonic mean of precision and recall ; beta > 1 gives more weight to recall, while beta < 1 favors precision.
With beta = 2, the recall is twice as important as the precision. You can see below the formula for the Fbeta score, with tp as True positives and fn as False negatives:

![image](https://github.com/user-attachments/assets/7777e62c-e097-4e75-b59c-9945d09779f9)

Besides the presence of motor, the goal was to predict its position. If the prediction was too far away from the ground truth, the prediction will be considered as a false negative. The challenge set a threshold of 1000 Å (angstrom, 10e-10m).

## Our implementation
To cope with the limited time accorded to the project (~10 days), we reduced the scope of the challenge by:
  - considering only tomograms with 0 or 1 motor
  - to ease preprocessing, we only considered tomograms with shape x<960, y<960 
  - we increased the prediction threshold from 1000 to 2000 Å

To answer the challenge we created 3 models:

![image](https://github.com/user-attachments/assets/da42082d-e96a-4359-b0c6-94d553cebf12)


The first two models used 2D images, obtained by creating a 'mean' image of the tomogram (mean pixel value accross all images of the tomogram) and use local equalieation thanks to the `exposure.equalize_adapthist` method from the scikit-image package to increase contrast.

We first predicted the presence of a motor in tomogram. We had a good fbeta score on this task: TODO mettre le fbeta score ici

Then we predicted the x, y position of the motor within the tomogram. We had a great decrease in our fbeta score, as it fell to : TODO mettre le fbeta score ici. There are two main reasons for a failed prediction of the position: either the distance between ground truth and prediction is greater than the threshold (TODO dire %), or the regression model can't find the class corresponding to the motor (TODO détailler ce point, @Pierre)

Then on the tomograms were we predicted x, y we use a third model to which we gave 3D data to obtain the z position, i.e. the slice where the motor was. We didn't loose much score at this step, as we ended with a fbeta score of: TODO mettre le fbeta score ici

The path worth exploring is to predict z before x,y ; indeed we may have a be better at this first task, and we could then try to predict x, y on the predicted slice or a set of slice centered on the predicted slice (to get more information).



## Contributors
<a href="https://github.com/mc-will/BYU_Locating-Bacterial_Flagellar_Motors_2025/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mc-will/BYU_Locating-Bacterial_Flagellar_Motors_2025" />
</a>
