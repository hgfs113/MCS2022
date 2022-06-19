# MCS2022.Car models verification competition


## TODO:
-   More augmentations
-   Test ArcFace, SphereFace losses
-   Test different architectures (models/models.py)

## Steps for working with baseline
### 0. Download CompCars dataset
To train the model in this offline the CompCars dataset is used. You can download it [here](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html).

### 0.5. Ensure you install all requirements
```bash
pip install -r requirements.txt
```

### 1. Prepare data for classification
Launch `prepare_data.py` to crop images on bboxes and generate lists for training and validation phases.
```bash
python prepare_data.py --data_path ../CompCars/data/ --annotation_path ../CompCars/annotation/
```

### 2. Run model training
```bash                         
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/baseline_mcs.yml
```
### 3. Create a submission file

```bash
CUDA_VISIBLE_DEVICES=0 python create_submission.py --exp_cfg config/baseline_mcs.yml \
                                                   --checkpoint_path experiments/baseline_mcs/model_0077.pth \
                                                   --inference_cfg config/inference_config.yml
```