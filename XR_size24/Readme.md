### Train Moco (main_moco.py)

```sh
# RB3D
python main_moco_nonorm.py -a RB3D --lr 0.003 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 2 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 3 --aug-plus ./Datasets/ 2>&1 | tee ./logs/Nonorm-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128.log

# DSRF3D_v2
python main_moco_nonorm.py -a DSRF3D_v2 --lr 0.00375 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 2 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 3 --aug-plus ./Datasets/ 2>&1 | tee ./logs/Nonorm-DSRF3D_v2_epochs600_bs32_lr0.00375_moco-k128_moco-dim128.log
```

### Visualizaiton (main_cluster.py)

The visualization results and the extracted numpy features will be saved in `./Figures`.

> Notes: As long as the extracted numpy features exist in the `./Figures` path, the features will not be re-extracted

- Visualizing the `train` datasets:

    ```sh
    # RB3D
    python main_cluster.py -a RB3D --batch-size 32 --workers 4 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128_checkpoint_0559.pth.tar --dist-url "tcp://localhost:10002" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/
    
    # DSRF3D_v2
    python main_cluster.py -a DSRF3D_v2 --batch-size 32 --workers 4 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-DSRF3D_v2_epochs600_bs32_lr0.00375_moco-k128_moco-dim128_checkpoint_0599.pth.tar --dist-url "tcp://localhost:10002" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/
    ```

- Visualizing the `val` datasets:

    ```sh
    # RB3D
    python main_cluster.py -a RB3D --batch-size 32 --workers 4 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128_checkpoint_0559.pth.tar --dist-url "tcp://localhost:10002" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --evaluate ./Datasets/
    
    # DSRF3D_v2
    python main_cluster.py -a DSRF3D_v2 --batch-size 32 --workers 4 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-DSRF3D_v2_epochs600_bs32_lr0.00375_moco-k128_moco-dim128_checkpoint_0599.pth.tar --dist-url "tcp://localhost:10002" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --evaluate ./Datasets/
    ```

### Label Spreading

```
numlabels = 20
```

**RB3D**

|       | Label Propagation | Label Spreading |
| :---: | :---------------: | :-------------: |
| Train |        0.2        |     0.9414      |
|  Val  |        0.2        |      0.951      |

**DSRF3D_v2**

|       | Label Propagation | Label Spreading |
| :---: | :---------------: | :-------------: |
| Train |        0.2        |     0.9563      |
|  Val  |        0.2        |      0.929      |

---

```
numlabels = 10
```

**RB3D**

|       | Label Propagation | Label Spreading |
| :---: | :---------------: | :-------------: |
| Train |        0.2        |     0.8261      |
|  Val  |        0.2        |      0.924      |

**DSRF3D_v2**

|       | Label Propagation | Label Spreading |
| :---: | :---------------: | :-------------: |
| Train |        0.2        |     0.8523      |
|  Val  |        0.2        |      0.925      |

---

```
numlabels = 5
```

**RB3D**

|       | Label Propagation | Label Spreading |
| :---: | :---------------: | :-------------: |
| Train |        0.2        |     0.6047      |
|  Val  |        0.2        |      0.951      |

**DSRF3D_v2**

|       | Label Propagation | Label Spreading |
| :---: | :---------------: | :-------------: |
| Train |        0.2        |      0.65       |
|  Val  |        0.2        |      0.912      |

### Train a linear classifier (main_lincls.py)

```sh
python main_lincls.py -a RB3D --lr 30 --schedule 150 --batch-size 32 --workers 8 --epochs 200 --pretrained ./main_moco_checkpoint/arch-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/ 2>&1 | tee ./logs/arch-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128_lincls_lr30.log

python main_lincls.py -a DSRF3D_v2 --lr 3 --schedule 30 60 90 --batch-size 32 --workers 8 --epochs 120 --pretrained ./main_moco_checkpoint/Nonorm-DSRF3D_v2_epochs600_bs32_lr0.003_moco-k128_moco-dim128.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/ 2>&1 | tee ./logs/Nonorm-DSRF3D_v2_epochs600_bs32_lr0.003_moco-k128_moco-dim128_lincls_lr3_epochs120.log
```

