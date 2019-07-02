# SiamFusion PyTorch implementation
## Introduction
This is my Thesis in the direction of Visual Object Tracking

## How to Run - Training
1. **Prerequisites:** The project was built using **python 3.6** and tested on Ubuntu 18.04 and 16.04. It was tested on a **GTX 1080 Ti**. Furthermore it requires [PyTorch 4.1](https://pytorch.org/).

2. Download the **GOT-10k** Dataset in http://got-10k.aitestunion.com/downloads and extract it on the folder of your choice, in my case it is `/home/arbi/desktop/GOT-10k` (OBS: data reading is done in execution time, so if available extract the dataset in your SSD partition).

3. In **config.py** script `root_dir_for_GOT_10k` change to your directory. 
```
root_dir_for_GOT_10k = '/home/arbi/desktop/GOT-10k' <-- change to your directory 
root_dir_for_VID     = ... (Optional for train on GOT-10k dataset)
root_dir_for_OTB     = ... (Optional for train on GOT-10k dataset) 
```

4. Run the **train.py** script:

```
python3 train.py
```
