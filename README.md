# ArUco Trajectory Extraction for UMI Project 

## 1 Usage

### 1.1 Configure pipeline
In the JSON format config file `./configs/pipeline_config.json`, you can configure the file paths of topdown and handheld camera calibration results (TODO), and decide whether to skip any of the processing steps.

### 1.2 Prepare files

#### 1.2.1 Images for camera calibration
- Capture images of 6x9 chessboard from topdown and handheld camera. 
- Put the images under directory `./configs/camera/<CAMERA_NAME>/images`, in which `<CAMERA_NAME>` is the name of the camera which will be used later in calibration.

#### 1.2.2 Videos for trajectory extraction

- Put the videos under directory `./videos/handheld` or `./videos/topdown`, depending on the camera used for recording.
- **NOTE THAT VIDEOS MUST BE ONE-TO-ONE MATCHED WHEN SORTED BY NAME.** This should be the case if the videos are recorded simultaneously, or you'll need to manually preprocess the videos to make them matched.

### 1.3 Run pipeline

Simply run the bash script `run_pipeline.sh` to start the processing pipeline.

```bash
bash run_pipeline.sh
```

When the pipeline is finished, you can find the dataset in `./output/dataset.pkl`. 

## 2 Test
### 2.1 Test pipeline
If you want to test whether the pipeline is working properly, you can:
- Prepare limited number of videos for testing, and put them under `./videos/handheld` or `./videos/topdown`.
- Run the test script `test_pipeline.sh`.
```bash
bash test_pipeline.sh
```
**NOTE**: In current version, running the bash script above will **OVERWRITE** the original `dataset.pkl` file. Make sure you have backed up the original dataset file if you want to keep it.

### 2.2 Test results
If you want to test whether the trajectory extraction results are correct, you can:
- Run the test script `test/test_dataset.py`.
```bash
python test/test_dataset.py
```

This script will load the dataset and plot the trajectories of the markers, and also generate a video that concatenate the original video and the trajectory plot. The videos of different demostration episodes should be located under directory `./output/dataset_videos`.