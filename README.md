# Keypoint Promptable Re-Identification
_Work in progress_

This project in build on top of [BPBreID](https://github.com/VlSomers/bpbreid).
Below are the instructions to download our proposed Occluded-PoseTrack Re-Identification benchmark and the annotations for four existing datasets.
The full codebase to reproduce the experiments will be released very shortly.

<p align="center">
  <img src="assets/gifs/001735_posetrack21.gif" width="600" /> 
</p>

## Download the Occluded-PoseTrack Re-Identification dataset
Our proposed dataset is derived from the [PoseTrack21](https://github.com/anDoer/PoseTrack21) dataset for multi-person pose tracking in videos.
The original PoseTrack21 dataset should be first downloaded following these [instructions](https://github.com/anDoer/PoseTrack21?tab=readme-ov-file#how-to-get-the-dataset).
We also provide json files that specify how the pose tracking annotations should be turned into our ReID dataset.
These json files describes which detections (bounding boxes + keypoints) should be used as train/query/gallery samples.
We provide these files and the related human parsing labels on [GDrive](https://drive.google.com/file/d/1n5rRx16D6Y9UpO-6nFD5yqAIYgIcfuPc/view?usp=sharing).
These files are read by our codebase to extract the ReID dataset from the pose tracking one and save the corresponding image crops on disk before launching the ReID experiment.
They can also be integrated in any external codebase in a similar maner.
The human parsing labels were generated using [SAM](https://github.com/facebookresearch/segment-anything) and [PifPaf](https://github.com/openpifpaf/openpifpaf), more details are provided in the [paper]().
More details are provided in the paper.


## Download annotations for extisting datasets
You can download the keypoint and human parsing labels on [GDrive](https://drive.google.com/drive/folders/15_RdnS1nr3iAYcnCibXmbT1LxU2n8PHZ?usp=sharing). 
The human parsing labels (.npy) were introduced by [BPBreID](https://github.com/VlSomers/bpbreid).
The keypoint annotations (.json) were generated with the [PifPaf](https://github.com/openpifpaf/openpifpaf) pose estimation model.
When multiple skeletons are detected within a single bounding box, the one with its head closer to the top center part of the image is considered as the ReID target, and marked with an 'is_target' attribute.
Around 10% of the query samples in the Occluded-Duke dataset were annotated manually because either the target person was not correctly labeled or the target person was not detected.
For Partiel-ReID, the first skeletons in the list is considered as the target.
For more detalsl, please have a look at the [paper]().
We provide the labels for five datasets: **Market-1501**, **Occluded-Duke**, **Occluded-ReID** and **Partia-ReID**.
After downloading, unzip the file and put the `masks` folder under the corresponding dataset directory.
For instance, Market-1501 should look like this:

    Market-1501-v15.09.15
    ├── bounding_box_test
    ├── bounding_box_train
    ├── external_annotation
    │   └── pifpaf_keypoints_pifpaf_maskrcnn_filtering
    │       ├── bounding_box_test
    │       ├── bounding_box_train
    │       └── query
    ├── masks
    │   └── pifpaf_maskrcnn_filtering
    │       ├── bounding_box_test
    │       ├── bounding_box_train
    │       └── query
    └── query

Make also sure to set `data.root` config to your dataset root directory path, i.e., all your datasets folders (`Market-1501-v15.09.15`, `Occluded_Duke`, `Occluded_REID`, `Partial_REID`) should be under this path.


## Download the pre-trained models
We also provide some [state-of-the-art pre-trained models](https://drive.google.com/drive/folders/1t4wXc2c3qlFaqUCifAlc_OPrFwvb7peD?usp=sharing) based on the Swin backbone.
You can put the downloaded weights under a 'pretrained_models/' directory or specify the path to the pre-trained weights using the `model.load_weights` parameter in the `yaml` config.
The configuration used to obtain the pre-trained weights is also saved within the `.pth` file: make sure to set `model.load_config` to `True` so that the parameters under the `model.bpbreid` part of the configuration tree will be loaded from this file.
