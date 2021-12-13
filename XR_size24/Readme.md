### Train Moco (main_moco.py)

```sh
# RB3D
python main_moco_nonorm.py -a RB3D --lr 0.003 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 2 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 3 --aug-plus ./Datasets/ 2>&1 | tee ./logs/Nonorm-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128.log

# DSRF3D_v2
python main_moco_nonorm.py -a DSRF3D_v2 --lr 0.00375 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 2 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 3 --aug-plus ./Datasets/ 2>&1 | tee ./logs/Nonorm-DSRF3D_v2_epochs600_bs32_lr0.00375_moco-k128_moco-dim128.log
## 10classes
python main_moco_nonorm.py -a DSRF3D_v2 --lr 0.003 --schedule 200 300 400 --batch-size 512 --moco-k 2048 --moco-dim 128 --workers 0 --epochs 500 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 1 --aug-plus ./Datasets/ 2>&1 | tee ./logs/Nonorm-DSRF3D_v2_epochs600_bs32_lr0.00375_moco-k128_moco-dim128-10classes.log

# YOPO
python main_moco_nonorm.py -a YOPO --lr 0.003 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 2 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 3 --aug-plus ./Datasets/ 2>&1 | tee ./logs/Nonorm-YOPO_epochs600_bs32_lr0.003_moco-k128_moco-dim128.log

python main_moco_nonorm.py -a YOPO --lr 0.003 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 2 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 3 --aug-plus --real ./Datasets/ 2>&1 | tee ./logs/Nonorm-YOPO_epochs600_bs32_lr0.003_moco-k128_moco-dim128-real.log
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
    
    # YOPO
    python main_cluster.py -a YOPO --batch-size 32 --workers 4 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-YOPO_epochs600_bs32_lr0.003_moco-k128_moco-dim128_checkpoint_0599.pth.tar --dist-url "tcp://localhost:10002" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/
    ```

- Visualizing the `val` datasets:

    ```sh
    # RB3D
    python main_cluster.py -a RB3D --batch-size 32 --workers 4 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128_checkpoint_0559.pth.tar --dist-url "tcp://localhost:10002" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --evaluate ./Datasets/
    
    # DSRF3D_v2
    python main_cluster.py -a DSRF3D_v2 --batch-size 32 --workers 4 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-DSRF3D_v2_epochs600_bs32_lr0.00375_moco-k128_moco-dim128_checkpoint_0599.pth.tar --dist-url "tcp://localhost:10002" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --evaluate ./Datasets/
    
    # YOPO
    python main_cluster.py -a YOPO --batch-size 32 --workers 4 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-YOPO_epochs600_bs32_lr0.003_moco-k128_moco-dim128_checkpoint_0599.pth.tar --dist-url "tcp://localhost:10002" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --evaluate ./Datasets/
    ```

### Clustering

**RB3D**

> The R-sqared value of Train Set is:  0.9528333333333333
>
> Variance explained per principal component: [0.57503784 0.11737645 0.08921021 0.06309234]
>
> The R-sqared value of Val Set is:  0.924
>
> Variance explained per principal component: [0.55872804 0.13060367 0.08781484 0.06282765]

**DSRF3D_v2**

> The R-sqared value of Train Set is:  0.8719444444444444
>
> Variance explained per principal component: [0.5507828  0.0764118  0.07582378 0.05093344]
>
> The R-sqared value of Val Set is:  0.744
>
> Variance explained per principal component: [0.54113233 0.08823619 0.07055311 0.05528075]

**YOPO** (5 Classes)

> the shape of extracted features for train set is:  (9000, 128)
>
> 
>
> The R-sqared value of train is:  0.9486111111111111
>
> The Accuracy of train is:  0.9888888888888889
>
> The label spreading score of train with 20 labeled samples each class is:  0.9885555555555555
>
> The label spreading score of train with 10 labeled samples each class is:  0.9643333333333334
>
> The label spreading score of train with 5 labeled samples each class is:  0.7608888888888888
>
> Variance explained per principal component: [0.35678694 0.09071694 0.0732204  0.0415383]
>
> 
>
> the shape of extracted features for val set is:  (1000, 128)
>
> The R-sqared value of val is:  0.8785000000000001
>
> The Accuracy of val is:  0.976
>
> The label spreading score of val with 20 labeled samples each class is:  0.947
>
> The label spreading score of val with 20 labeled samples each class is:  0.911
>
> The label spreading score of val with 20 labeled samples each class is:  0.961
>
> Variance explained per principal component: [0.3496042  0.09186376 0.07745937 0.04443906]

**YOPO** (10 Classes)

> the shape of extracted features for train set is:  (18000, 128)
>
> Variance explained per principal component: [0.8017566  0.06142639 0.03949744 0.01038518]
>
> The label spreading score of train with 20 labeled samples each class is:  0.7004444444444444
>
> The label spreading score of train with 10 labeled samples each class is:  0.6699444444444445
>
> The label spreading score of train with 5 labeled samples each class is:  0.6141111111111112
>
> 
>
> the shape of extracted features for val set is:  (2000, 128)
>
> Variance explained per principal component:[0.80836123 0.05891716 0.03839895 0.01042624]
>
> The label spreading score of val with 20 labeled samples each class is:  0.8125
>
> The label spreading score of val with 10 labeled samples each class is:  0.77
>
> The label spreading score of val with 5 labeled samples each class is:  0.7315
>
> 

**YOPO** (Real)

> Variance explained per principal component: [0.31175312 0.1309359  0.07007271 0.05420246]

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

