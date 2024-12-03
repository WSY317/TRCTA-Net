## Installation
```shell
#Step 1: Create a new pyskl environment
conda create -n pyskl python=3.8
#Step 2: Install PyTorch version 1.10.1 with CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
#Step 3: Install related packages in pyskl, modify requirements.txt to use mmcv-full==1.5.0
cd pyskl
pip install -r requirements.txt
pip install -e .
#Step 4: Install mmpose and mmdet, change to the TRCTA directory
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
#Step 5: Install additional dependencies
pip install ftfy
pip install regex
pip install pandas
```

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```

## Demo
You can use the following commands in the demo folder along with the pre-trained models we provide to test the model performance. Before running the skeleton action recognition demo, make sure you have installed. We recommend you to directly use the provided conda environment, with all necessary dependencies included:mmcv-full mmpose and mmdet.
```shell
python demo/demo_skeleton.py demo/waving.avi demo/waving.mp4 --checkpoint {checkpoint}
python demo/demo_skeleton.py demo/brushing_teeth.avi demo/brushing_teeth.mp4 --checkpoint {checkpoint}
```