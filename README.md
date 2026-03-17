# ENVIRONMENT SETUP

## 1.Make conda environment
```bash
conda create -n ifblend python=3.8
conda activate ifblend
```
## 2.Install dependencies and clone the repository
```bash
git clone https://github.com/CeviKle/NTIRE2026-KLETech-CEVI-Denoising.git
cd NTIRE2026-KLETech-CEVI-Denoising
```
## 3.Install PyTorch
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
## 4.Install required dependencies
```bash
pip install -r requirements.txt
```
## 5.Install BasicSR
```bash
python setup.py develop --no_cuda_ext
```
Download the pretrained weights from [Google Drive](https://drive.google.com/file/d/16sN3kjzhbNJyJzKW_Ruq8BDYFUCzAl0g/view?usp=drive_link) and put it in **model_zoo folder**

## RUN TESTING

To run the test code for Gaussian color image denoising (σ = 50), use the following command:
```bash
python basicsr/test.py -opt options/test/test_swinir.yml
```
