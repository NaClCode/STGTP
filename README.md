# Uncertainty-aware ship trajectory prediction via Spatio-Temporal Graph Transformer

<p align="center">
    <img src="https://img.shields.io/static/v1?label=Python&message=3.8&color=e83e8c"/>
    <img src="https://img.shields.io/static/v1?label=Pytorch&message=2.0.0&color=fd7e14"/>
    <img src="https://img.shields.io/static/v1?label=CUDA&message=11.8&color=007bff"/>
</p>

Code for [Uncertainty-aware ship trajectory prediction via Spatio-Temporal Graph Transformer](https://www.sciencedirect.com/science/article/pii/S1366554525003564)

## Run

```
pip install -r requirements.txt
python trainval.py --dataset <your dataset> --model STGTP
```

## Dataset

Please prepare the dataset in the following format:

### Dataset Directory

```
├─data
    ├─<your dataset 1>
        |─test
            |─ <your dataset 1>.csv
        |─train
            |─ <your dataset 1>.csv
        |─val
            |─ <your dataset 1>.csv
```

### Dataset Format \[Example]

| Time | MMSI | Latitude | Longitude |
| ---- | ---- | --- | --- |
|   0   |  111111111    |   39.90  |  116.40    |

### Cite STGTP
If you find this repo useful, please consider citing paper

```
@article{gong2025uncertainty,
  title={Uncertainty-aware ship trajectory prediction via Spatio-Temporal Graph Transformer},
  author={Gong, Jincheng and Li, Huanhuan and Jiao, Hang and Yang, Zaili},
  journal={Transportation Research Part E: Logistics and Transportation Review},
  volume={203},
  pages={104315},
  year={2025},
  publisher={Elsevier}
}
```

### Reference
The code base heavily borrows from [STAR](https://github.com/cunjunyu/STAR)