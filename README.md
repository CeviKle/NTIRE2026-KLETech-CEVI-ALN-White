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
* Download the <code>checkpoints.zip</code> from [here](https://drive.google.com/drive/folders/1gwY7ZjE4Uzwj2q7FlP6hWUf_TR0zXKg8?usp=drive_link) and unzip to the repository root directory. 
* Download the <code>weights.zip</code> from [here](https://drive.google.com/drive/folders/1gwY7ZjE4Uzwj2q7FlP6hWUf_TR0zXKg8?usp=drive_link) and unzip in the root directory of the repository.
* Download the <code>results</code> from [here](https://drive.google.com/drive/folders/1ZUuehtFrICEwFv-K0DMAQuIY_86ErVwH?usp=drive_link)

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
