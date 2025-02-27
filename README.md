- pip install git+https://github.com/TorchDSP/torchsig.git
- pip install git+https://github.com/facebookresearch/detectron2

This command should run

```bash
python tools/pretraining.py --num-epochs 10 --to-float-32 true --backbone resnet50 --backbone-provider timm  --online-linear-eval true
```

Todos for this project:

- fix transforms
- update number of classes in vitdet
- vitdet evaluation
