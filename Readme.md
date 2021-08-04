### Train Moco (main_moco.py)

```
python main_moco.py -a RB3D --lr 0.003 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 4 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 2 --aug-plus ./Datasets/ 2>&1 | tee ./logs/arch-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128.log

python main_moco.py -a DSRF3D_v2 --lr 0.03 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 4 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --aug-plus ./Datasets/ 2>&1 | tee ./logs/arch-DSRF3D_v2_epochs600_bs32_lr0.03_moco-k128_moco-dim128.log
```

### Visualizaiton (main_cluster.py)

The visualization results and the extracted numpy features will be saved in `./Figures`.

> Notes: As long as the extracted numpy features exist in the `./Figures` path, the features will not be re-extracted

- Visualizing the `train` datasets:

    ```
    python main_cluster.py -a DSRF3D_v2 --lr 0.3 --batch-size 32 --workers 8 --epochs 100 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-DSRF3D_v2_epochs120_bs32_lr0.3_moco-k128_moco-dim128.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/
    ```

- Visualizing the `val` datasets:

    ```
    python main_cluster.py -a DSRF3D_v2 --lr 0.3 --batch-size 32 --workers 8 --epochs 100 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-DSRF3D_v2_epochs120_bs32_lr0.3_moco-k128_moco-dim128.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --evaluate ./Datasets/
    ```

### Train a linear classifier (main_lincls.py)

```
python main_lincls.py -a RB3D --lr 300 --schedule 150 --batch-size 32 --workers 8 --epochs 200 --pretrained ./main_moco_checkpoint/bs32_lr0.003_moco-k128_moco-dim256.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/ 2>&1 | tee ./logs/arch-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim256_lincls_lr300.log
```

