## structural language models of code 
###### reproduction

#### paper
https://arxiv.org/pdf/1910.00577.pdf

#### fit on google.colab
```
!git clone https://github.com/tihonovcore/model.git

!python model/fit.py

!zip -r /content/weights.zip /content/model/saved_model
from google.colab import files
files.download("/content/weights.zip")
```

#### predict on train dataset 
```
!python model/predict.py --json_path=/content/model/dataset/dataset.json
```
