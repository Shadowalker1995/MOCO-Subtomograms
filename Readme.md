### Train Moco

```
python main_moco.py -a RB3D --lr 0.003 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 512 --workers 4 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 2 --aug-plus ./Datasets/ 2>&1 | tee ./logs/arch-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim256.log

python main_moco.py -a DSRF3D_v2 --lr 0.03 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 4 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --aug-plus ./Datasets/ 2>&1 | tee ./logs/arch-DSRF3D_v2_epochs600_bs32_lr0.03_moco-k128_moco-dim128.log
```



### Train a linear classifier

```
python main_lincls.py -a RB3D --lr 300 --schedule 150 --batch-size 32 --workers 8 --epochs 200 --pretrained ./main_moco_checkpoint/bs32_lr0.003_moco-k128_moco-dim256.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/ 2>&1 | tee ./logs/arch-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim256_lincls_lr300.log
```

