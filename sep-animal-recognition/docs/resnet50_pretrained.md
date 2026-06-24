# ResNet-50 pretrained fine-tuning

Bu deney, torchvision ImageNet ağırlıklarıyla başlatılan ResNet-50'nin bütün
katmanlarını 21 çıktı için fine-tune eder. Mevcut scratch modelleri ve ortak
eğitim scriptleri değiştirilmez.

## Dosyalar

- `configs/resnet50_pretrained.json`
- `scripts/train_resnet50_pretrained.py`
- `slurm/train_resnet50_pretrained.sbatch`

## Yerel kullanım

`sep-animal-recognition` klasöründen:

```bash
export PYTHONPATH="$PWD/src"
python scripts/train_resnet50_pretrained.py \
  --config configs/resnet50_pretrained.json \
  --device auto
```

## SLURM kullanımı

```bash
sbatch slurm/train_resnet50_pretrained.sbatch
```

Script, `pretrained: true` alanını zorunlu tutar ve
`ResNet50_Weights.DEFAULT` ağırlıklarını yükler. İlk çalıştırmada ağırlıkların
torchvision cache'ine indirilmesi gerekebilir. En iyi checkpoint validation
macro-F1 değerine göre `runs/resnet50_pretrained/best.pt` olarak kaydedilir.
