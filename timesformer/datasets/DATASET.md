# Dataset Preparation

## Kinetics

The Kinetics Dataset could be downloaded from the following [link](https://github.com/cvdfoundation/kinetics-dataset):

After all the videos were downloaded, resize the video to the short edge size of 256, then prepare the csv files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv`. The format of the csv file is:

```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```

## Something-Something V2
1. Please download the dataset and annotations from [dataset provider](https://20bn.com/datasets/something-something).

2. Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv)).

3. Extract the frames at 30 FPS. (We used ffmpeg-4.1.3 with command
`ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"`
   in experiments.) Please put the frames in a structure consistent with the frame lists.

Please put all annotation json files and the frame lists in the same folder, and set `DATA.PATH_TO_DATA_DIR` to the path. Set `DATA.PATH_PREFIX` to be the path to the folder containing extracted frames.
