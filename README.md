# pytorch-Lane Detection

Code for paper 'Semi-Supervised Domain Adaptation with Dual-Adversarial Learning for Lane Detection'

### Introduction
This is a PyTorch(1.10.0) implementation of [LD-SSDA](https://github.com/shenhuqiji/LD-SSDA). 
The code was tested with Anaconda and Python 3.8.8.
```Shell
    conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### Installation

After installing the dependency:    
``` Shell
    pip install pyyaml
    pip install pytz
    pip install tensorboardX==1.4 matplotlib pillow 
    pip install tqdm
    conda install scipy==1.1.0
    conda install -c conda-forge opencv
```


1. Configure your dataset path in [train.py] with parameter '--data-dir'.
    Dataset download link: 
        [VIL100](https://drive.google.com/drive/folders/178_SSeQ4M1hI3BrTonhiTrpOWTEAenLE)
        [Tusimple](https://github.com/TuSimple/tusimple-benchmark/)
        

2. You can train model using ERFNet, deeplab v3+ or other backbones.

    To train it, please do:
    ```Shell
    python train.py -g 0 --data-dir /dataset --batch-size 8 --datasetT VIL100 --datasetS Tusimple
    ```
    To test it, please do:
    ```Shell
    python test.py --model-file ./checkpoint.pth.tar
    ```

