# ENVIRONMENT SETUP

## 1.Make conda environment
```bash
conda create -n ifblend python=3.10
conda activate ifblend
```
## 2.Install dependencies and clone the repository
```bash
git clone https://github.com/CeviKle/NTIRE2026-KLETech-CEVI-ALN-White.git
cd NTIRE2026-KLETech-CEVI-ALN-White
```
## 3.Install required dependencies
```bash
pip install -r requirements.txt
```
* Download the <code>checkpoints.zip</code> from [here](https://drive.google.com/file/d/1nNY1vF7mwVRWgTtdhJRmF5s9yGCAMZfm/view?usp=sharing) and unzip to the repository root directory. 
* Download the <code>weights.zip</code> from [here](https://drive.google.com/file/d/1rwv2G8tAboGEzEsczMMiSMzxCA1to3f7/view?usp=sharing) and unzip in the root directory of the repository. 

## RUN TESTING

To run the test code for Ambient Lightining Normilization, use the following command:
```bash
python eval.py \
--model_name ifblend \
--data_src /NTIRE2026/C2_ALN_White/ntire26_aln_test_in \
--res_dir ./final-results \
--ckp_dir ./checkpoints \
--load_from IFBlend_ALN
```
