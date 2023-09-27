###  Mammogram ROI Classification using Multi-Agent Reinforcement Learning. Trained on mass images on CBIS-DDSM dataset 

#### Installation
```bash
$ cd /path/to/MARLClassification
$ # create and activate your virtual env
$ python -m venv venv
$ ./venv/bin/activate
$ # install requirements
$ pip install -r requirements.txt
```

#### Usage
To run training :
Put the (CBIS-DDSM)[https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset] dataset folder in 
./resources/downloaded/cbis and run
```bash
$ python -m marl_classification -a 16 --step 32 --cuda --run-id train_cbis train --action [[5,0],[-5,0],[0,5],[0,-5]] --img-size 224 --nb-class 2 -d 2 --f 24 --ft-extr cbis --nb 256 --na 256 --nm 64 --nd 32 --nlb 256 --nla 256 --batch-size 32 --lr 1e-4 --nb-epoch 100 --eps 1.0 --eps-dec 0.99995 -o ./out/cbis
```

Forked from: https://github.com/Ipsedo/MARLClassification