## Installation

firstly create the conda environment

```
conda create -n <env-name> python==3.10

```


then run

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install iopath

python3 -m pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'

pip3 install -r requirements.txt
```

You are on last step of installation hurry!!!

```
python3 setup.py install

```


# Dataset Strcuture

```
project_root/
├── Video_fold/
│   ├── Train_video/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── Valid_video/
│       ├── videoX.mp4
│       ├── videoY.mp4
│       └── ...
├── Metadata.csv
```


```video_name```: Name of the video file (must match the file in the folder).

```train/valid```: Specifies whether the sample is used for training (Train_video) or validation (Valid_video).

```classname```: The action or category label for the video.

# Download Checkpoint

Firslty you need to Download the checkpoint from the given command

```
wget -P checkpoints https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_L_40x3_k400_f306903192.pyth
```


# Finetuning

change the Config file according to you 

you need to change ```configs/Kinetics/MVITv2_L_40x3_test.yaml```

```
python3 train.py
```



