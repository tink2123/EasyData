# PP-EDA 快速开始

说明：本文主要介绍EasyData whl包对[PP-EDA](./EasyDataAug.md)工具的快速使用，如需使用数据质量提升相关功能，请参考教程[PP-LDI](../LDI/quick_start.md)

<a name="1"></a>
## 1. 安装

<a name="11"></a>
### 1.1 安装PaddlePaddle

- 您的机器安装的是CUDA9或CUDA10，请运行以下命令安装

  ```bash
  python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```

- 您的机器是CPU，请运行以下命令安装

  ```bash
  python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

<a name="12"></a>
### 1.2 安装EasyData whl包

pip安装

```bash
pip install easydata
```

本地构建并安装

```bash
git clone https://github.com/PaddlePaddle/EasyData.git
python3 setup.py bdist_wheel
pip3 install dist/easydata-x.x.x-py3-none-any.whl # x.x.x是easydata的版本号
```

<a name="2"></a>
## 2. 便捷使用
<a name="21"></a>
### 2.1 命令行使用

EasyData提供了一系列demo测试图片，可以在终端中切换到相应目录

```
cd /path/to/demo
```

如果不使用提供的测试图片，可以将下方`--ori_data_dir`和`--label_file`参数替换为相应的测试图片路径和标签路径。

<a name="211"></a>
#### 2.1.1 图像分类模型

```bash
easydata --model ppeda --ori_data_dir demo/clas_data/ --label_file demo/clas_data/train_list.txt --model_config deploy/configs/ppeda_clas.yaml 
```
运行该命令后，会输出增广后的图像和标签文件，增广图像路径默认在test目录下，该目录下有对应的增广子文件夹，如下所示：

```
├── test                                
│   ├── randaugment    
│   ├── random_erasing    
│   ├── gridmask    
│   ├── tia_distort    
│   ├── tia_stretch    
│   ├── tia_perspective    
```

增广后的有效标签位于high_socre_label.txt，内容如下：
```
./test/randaugment/1_n01440764_15008.JPEG 0
./test/randaugment/2_n01530575_10039.JPEG 1
./test/randaugment/3_n01601694_4224.JPEG 2
./test/randaugment/4_n01641577_14447.JPEG 3
./test/randaugment/5_n01682714_8438.JPEG 4
```
   

<a name="212"></a>
#### 2.1.2 文本识别模型

```bash
easydata --model ppeda --ori_data_dir demo/ocr_data/ --label_file demo/ocr_data/train_list.txt --model_config deploy/configs/ppeda_ocr.yaml --model_type ocr_rec

```

<a name="213"></a>
#### 2.1.3 图像识别模型

```bash
easydata --model ppeda --ori_data_dir demo/shitu_data --label_file demo/shitu_data/train_list.txt --model_config deploy/configs/ppeda_shitu.yaml
```

<a name="22"></a>
### 2.2 Python脚本使用
easydata不仅支持命令行使用，还支持Python脚本进行调用使用，以图像分类模型为例:

```python
from easydata import EasyData

ppeda = EasyData(model='ppeda', ori_data_dir='demo/clas_data', label_file='demo/clas_data/train_list.txt', model_config='deploy/configs/ppeda_clas.yaml')
ppeda.predict()
```
如需更换其他场景，可以修改model_config字段，具体参数说明可以参考第三节

<a name="3"></a>

## 3. 参数说明
| 字段 | 任务类型 |说明 | 默认值 |
|---|---|---|---|
| model | 通用 |使用的模型工具，可选ppeda,ppldi | ppeda |
| gen_mode | 数据生成 | 数据生成类型，可选 aug，ocr_rec | aug |
| model_type | 数据生成 | 场景模型类型，可选 cls, ocr_rec | cls |
| model_config | 数据生成 | 使用的场景模型配置 | deploy/configs/ppeda_clas.yaml |
| ori_data_dir | 数据生成 | 原始数据目录 | None |
| label_file | 数据生成 | 原始数据标签 | None |
| gen_label | 数据生成 | 增广后的数据标签 | labels/test.txt |
| img_save_folder | 数据生成 | 增广图像存储目录 | test |
| size | 数据生成 | 输出图像尺寸 | 224 |
| gen_num | 数据生成 | 每类增广数量 | 10 |
| gen_ratio | 数据生成 | 每类增广倍数，优先级低于gen_num；如果gen_num大于原始数据量，该参数生效 | 0 |
| ops | 数据生成 | 增广op | "randaugment", "random_erasing", "gridmask", "tia_distort", "tia_stretch", "tia_perspective" |
| bg_num_per_word | 数据生成 | 每条OCR语料选择几张背景图 | 5 |
| bg_img_dir | 数据生成 | 文本图像生成背景图目录 | demo/ocr_rec/bg |
| font_dir | 数据生成 | 文本图像生成字体目录 | demo/ocr_rec/font |
| corpus_file | 数据生成 | 文本图像生成语料路径 | demo/ocr_rec/corpus.txt |
| threads | 数据生成 | 文本图像线程数 | 1 |
| repeat_ratio | 数据去重 | 图像去重的阈值，图像相似度得分大于该阈值会被剔除 | 0.9 |
| compare_out | 数据去重 | 去重过滤的中间结果 | tmp/rm_repeat.txt |
| quality_ratio | 低质过滤 | 低质过滤的阈值，图像质量得分低于该阈值会被剔除 | 0.2 |
| final_label | 低质过滤 | 最终生成的有效数据标签 | high_socre_label.txt |


<a name="4"></a>

## 4. 小结

通过本节内容，相信您已经掌握了EasyData whl包的使用方法并获得了初步效果。
