# Overview
## TASK

Participants are invited to develop algorithms for the automated segmentation of solar filaments. Any computational approach may be employed, including traditional image-processing techniques, machine learning methods, and state-of-the-art deep neural network architectures. The objective of this challenge is to develop an algorithm---whether novel or existing---that generates accurate segmentation masks for individual solar filaments. For each filament, the predicted mask should capture the complete extent of the filament, including its fine-scale structures, while minimizing the inclusion of non-filament regions. The challenge emphasizes robust and precise delineation of filament morphology across a diverse set of solar observations.

## GOAL

This competition focuses on the pixel-level precision of filament segmentation masks. The primary evaluation criterion is the Dice score, which quantifies the degree of overlap between the predicted and ground-truth segmentation masks (see [torchmetrics.segmentation.DiceScore](https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html)). In addition to segmentation accuracy, the evaluation framework incorporates penalties related to fragmentation and over-segmentation (e.g., one-to-many and many-to-one correspondences between predicted and ground-truth filament structures), as well as the end-to-end computational efficiency of the proposed method.


#### Important Dates

- July 10, 2026: Competition is launched. Contestants can start working on the challenge.
- Nov 15, 2026: Deadline for contest teams to submit final reports and solutions.
- Nov 30, 2026: The winning teams will be announced.
- Dec 14-17, 2026: (Optional) IEEE BigData conference will take place at Phoenix, AZ. The winners of this BigDataCup competition (as well as all other BigDataCups) will be announced

## Description
#### What Are Solar Filaments?

Filaments (as shown below) are dense "clouds" of solar material suspended by magnetic field lines above photospheric neutral lines. The significance of filaments for space weather research lies in the fact that they are at the core of solar eruptions, including Coronal Mass Ejections (CMEs), solar flares, and Solar Energetic Particle (SEP) storms. An Earth-directed CME can cause enormous damage to the electric power grid, disrupt GPS systems, create radiation hazards for passengers and crew on polar flights, and be lethal to astronauts traveling outside the protective bubble provided by the Earth's magnetosphere. This is why solar filaments are a critical event type in space weather research and forecasting operations.

![An example of a solar filament](inbox_3070992_77fe20a042912564b30d941836d26f70_filament_GONG_20240103224642Ch.png)

#### What Is MAGFiLO?

MAGFiLO, short for Manually Annotated GONG Filaments from H-Alpha Observations, is the ground-truth data of filaments' segmentation mask ([10.1038/s41597-024-03876-y](https://www.nature.com/articles/s41597-024-03876-y)). It is created for training and evaluation of machine-learning algorithms, and for evaluation of traditional computer-vision algorithms.

The illustration below shows an H-Alpha observation of the Sun, in which solar filaments have been identified by expert human annotators. Although the dataset includes additional annotations, such as bounding boxes, filament spines, and class labels, this competition focuses exclusively on the segmentation masks of solar filaments.

MAGFiLO is the testbed for all participants' models.

See the Data tab for more details.

![An example of a MAGFiLO instances](inbox_3070992_7b19d336400715aa6b69a895f6cee0ed_magfilo_segmentation_examples.png)

#### What Are the Main Challenges?

Although recent advances in object segmentation have led to remarkable performance across many domains, solar filament segmentation remains a challenging task for several reasons:

- **Fine-scale structures.** Accurately capturing fine filament structures, such as barbs, remains difficult. Barbs are thin, thread-like features that extend from the main body of a filament along a characteristic orientation. Their orientation relative to the filament spine carries important information about the filament's underlying magnetic field configuration.

- **Background noise and image quality.** Distinguishing filament material (dark regions) from background structures and noise is nontrivial. Since the observations used in this competition originate from ground-based observatories, they are affected by various sources of noise and imaging artifacts. Accurately identifying small-scale filament structures while suppressing background noise remains a significant challenge.

- **Structural continuity.** Existing segmentation algorithms often struggle to identify solar filaments as contiguous physical structures. Instead, they may produce fragmented segmentations or segment clusters of nearby dark regions ("islands"), resulting in incomplete or physically inconsistent representations of the filament morphology.

## Evaluation

All submissions are judged based on the following rubric:

- Quantitative Comparison (70%):
    - Mean Dice score (using [torchmetrics.segmentation.DiceScore](https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html))
    - Distribution of Dice scores
    - Distribution of IoU scores
    - Distribution of one-to-many and many-to-one relations between the ground-truth and predicted segmentations.
- Qualitative Comparison (30%):
    - Detailed description of the entire pipeline (from preprocessing to final prediction, including the architecture of the utilized algorithm).
    - The apparent morphology of predicted segmentations on H-Alpha images.
    - The quality of code (modularity and documentation)

**Note:** The evaluation of the above criteria is contingent upon the availability of the source code. Please refer to the Open-Access Policy section below.

## Submission File

We expect the participants to upload a single CSV file for the entire test set, where each row corresponds to one predicted filament, as shown below.

filament_id        | segmentation_rle   |
-------------------|--------------------|
20150125172714Mh_1 | "f8uSDds ... VQNC" |
20150125172714Mh_2 | "KHT%$HD ... 9>km" |
20150125172714Mh_3 | "YQNEgn1 ... BH6^" |
...                | ...                |
20170501024112Bh_1 | "HBy4d6D ... 97*D" | 

- Column `filament_id` contains unique ids for each filament, e.g., `20150125172714Mh_2`.

- Column `segmentation_rle` contains RLE counts each encoding the mask corresponding to one filament, e.g., ``^Vj02jo16I5O2O1`PNA]o1c0N19G1N11O3L01O4JYamT3``. Do not include quotations (`'` or `"`).

**Note:** You only need to store RLE Counts; no need to store RLE Size. The Size is fixed for all images: 2048 X 2048 pixels.

To convert your output (e.g., masks, polygons, etc.) to the expected format (RLE Counts), please use the `pycocotools` [Python package](https://pypi.org/project/pycocotools/), specifically, the methods `annToMask`, `decodeMask`, and `encodeMask`. For more details, see [coco.py](https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/coco.py).

**Note:** The tail strings in the `filament_id` column (e.g., `_2`) is only to make the rows unique. For example, if your algorithm identifies 3 filaments in a given image with id `20150125172714Mh`, your should generate 3 rows with the following keys: `20150125172714Mh_1`, `20150125172714Mh_2`, and `20150125172714Mh_3`. As long as the tail strings render the rows unique , and the image id remains unchanged, your filament id is acceptable.
Note: The number of ground-truth segmentations may also be different from the number of predicted ones. So, the evaluation method matches the predicted and ground-truth segmentations based on their actual overlap, not their index.

The scoreboard will show to the contestants the mean Dice score for ~50% of images in the test set. The results on the remaining images will be only visible to the organizers of the competition. For a fair evaluation pipeline, the two scores should be fairly close.
