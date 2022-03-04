# Skeleton Anonymizer
## Install Dependecies
```
pip install -r requirements.txt
```

## Dataset
You can request and download the datasets here:
* [ETRI](https://nanum.etri.re.kr/share/judekim/HMI4?lang=En_us)
* [NTU](https://rose1.ntu.edu.sg/dataset/actionRecognition/).

Use `data_gen/etri_data.py` and `data_gen/etri_data_split.py` to preprocess ETRI data, and use `data/gen_ntu_gendata.py` to preprocess NTU data.

## Pretraining
`python pretrain_classifier --config config/config_file.yaml --work-dir work_dir/work_dir`

`config_file.yaml` is one of `etri_action.yaml`, `etri_gender.yaml`, `ntu_action.yaml`, and `ntu_reid.yaml`.

`work_dir` is the directory that the results are saved.

## Training
`python train_anonymizer.py --config config/config_file.yaml --work-dir work_dir/work_dir`

`work_dir` is the directory that the results are saved.

## Acknowledgements
The code is based on [MS-G3D](https://github.com/kenziyuliu/MS-G3D).
Thanks to the original authors!
