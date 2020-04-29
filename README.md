# 分类器构建

本项目使用KNN算法来构建分类器。

## 使用

1. 先将`data`文件夹下的数据集进行划分:

```shell script
python src/model/divider.py
```

2. 对参数进行优化

```shell script
python src/model/core.py \
  --do_optimize
```

3. 评估模型效果

```shell script
python src/model/core.py \
    --do_eval
```

# 实验结果

权值未优化（六个特征的权值全为1）：`76.3%`

权值优化之后(实验结果待定)：`82.2%`
