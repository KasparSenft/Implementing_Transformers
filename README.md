## Implementing a Transformer Architecture for Translation from Scratch

#### To set up the environment 

```shell
conda create -n imptransf python==3.11.0
conda activate imptransf
pip install -r requirements.txt
```

#### To download and clean the dataset run:

```
python dataset.py
```

#### To Train Model:

```
python main.py --d_model=128 --num_heads=8 --num_layers=6 --epochs=50 --dim_feed_forward=512
```