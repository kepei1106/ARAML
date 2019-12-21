# ARAML: A Stable Adversarial Training Framework for Text Generation

## Introduction

Adversarial reward augmented maximum likelihood (ARAML) is an adversarial training framework to deal with the instability issue of training text GANs. You can read our [paper](https://www.aclweb.org/anthology/D19-1436.pdf) for more details. This project is a TensorFlow implementation of our work.

## Dependencies

* Python 2.7
* NumPy
* SciPy
* TensorFlow >= 1.3.0

## Quick Start

* Dataset

  Our experiments contain three datasets, i.e. COCO, EMNLP2017 WMT and WeiboDial. You can find them in the */data* directory.

* Train

  For the COCO dataset,

  ```sh
  cd src/coco_emnlp
  python araml_lm.py --task_name coco
  ```

  Similarly, you can run the codes for the EMNLP dataset by changing the argument of *task_name* to emnlp.

  For the WeiboDial dataset,

  ```sh
  cd src/weibodial
  python main_adver_lmsample.py
  ```


## Details

### Hyper-parameter

You can set most of the hyper-parameters about the structure of the models in */src/coco_emnlp/conf_coco.py (conf_emnlp.py)* or */src/weibodial/utils/conf.py* for three datasets, respectively. As for WeiboDial, some of the hyper-parameters related to the training process can be set as the arguments. Refer to */src/weibodial/main_adver_lmsample.py* for more details.

### Word Vector

For COCO and EMNLP datasets, we follow the existing works on text GANs and use the randomly initialized word embedding. 

For WeiboDial, we adopt the pre-trained word embedding whose format is the same as [GloVe](https://nlp.stanford.edu/projects/glove/). You can use your own word vectors or initialize them randomly. Refer to *build_word2vec* function in */src/weibodial/utils/data_utils.py* for more details.

### Result

For COCO and EMNLP datasets, you can get the generated results in the files which begin with result (in */src/coco_emnlp/res_coco or res_emnlp*), evaler and cotra (in */src/coco_emnlp/log_coco or log_emnlp*). For example, the generated results in the *result_xxx* and *evaler_xxx* files contain token IDs as follows:

```
65 3867 274 4215 1863 4728 976 576 1173 185 1039 193 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814
65 4712 4218 1863 4068 3361 2606 1863 3126 2606 3277 193 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814 1814
```

The results in the *cotra_xxx* files include tokens corresponding to the *evaler_xxx* file:

```
A laptop lays on a table next with pies and glasses .
A view of a white sink in a house in mirror .
```

As for WeiboDial, you can directly check the generated results in */src/weibodial/gen_test*. Each sample consists of a post, a true response and a generated response:

	额 … 没事 没事 明天 陪 我 吃 就 好 了
	爱 死 你 了 ！
	好 啊 ！ 你 回来 请 你 吃 ！


## Citation

```
@inproceedings{ke-etal-2019-araml,
    title = "{ARAML}: A Stable Adversarial Training Framework for Text Generation",
    author = "Ke, Pei  and Huang, Fei  and Huang, Minlie  and Zhu, Xiaoyan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1436",
    doi = "10.18653/v1/D19-1436",
    pages = "4270--4280",
}
```

**Please kindly cite our paper if this paper and the codes are helpful.**

## Thanks

Many thanks to the GitHub repositories of [SeqGAN](https://github.com/ChenChengKuan/SeqGAN_tensorflow) and [IRL](https://github.com/FudanNLP/Irl_gen). Part of our codes are modified based on their codes.

## License

Apache License 2.0

