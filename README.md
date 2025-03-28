# ABIL: Learning for Long-Horizon Planning via Neuro-Symbolic Abductive Imitation

### [Paper](https://arxiv.org/abs/2411.18201) | [Project Page](https://www.lamda.nju.edu.cn/shaojj/KDD25_ABIL/)

## News
- **2024.12.30** Release code.
- **2024.12.22** Release training data.
- **2024.11.17** ABIL is accepted by **KDD 2025**!

## The Framework of Abductive Imitation Learning.

![ABIL](./images/framework.png)


## Contents

- [Install](#install)
- [Data](#data)
- [Train](#train)
- [Evaluation](#evaluation)

 🚧 This repository is under construction 🚧 -- Please check back for updates!
### Install

1. Clone the repo into a local folder.

```bash
git clone https://github.com/Hoar012/ABIL-KDD-2025.git

cd ABIL-KDD-2025
conda create -n ABIL python=3.8
conda activate ABIL
pip install -r requirements.txt
```

2. Clone the Jacinle repo.

```bash
git clone https://github.com/vacancy/Jacinle --recursive
export PATH=<path_to_jacinle>/bin:$PATH
```

#### mini-behavior environment
``` bash
cd ./hacl/envs/mini_behavior
pip install -e .
```

#### Cliport environment
```bash
cd ./hacl/envs/cliport
export CLIPORT_ROOT=$(pwd)
python setup.py develop
```


### Data
Our training demonstrations are generated by Python scripts. View them separately in the following files:

- BabyAI: `hacl/p/kfac/minigrid/data_generator.py`
- Mini-BEHAVIOR: `hacl/p/kfac/minibehavior/data_generator.py`
- CLIPort: `cliport_src/data_generator.py`


### Train

**BabyAI**

1. Train the grounding model.

```bash
jac-run babyai_src/train-babyai-abl.py minigrid goto  --use-offline=yes --structure-mode abl --action-loss-weight 1 --evaluate-interval 0 --iterations 1000 --append-expr
```

2. Train the Imitation Learning model.

```bash
jac-run babyai_src/babyai-abil-bc.py minigrid goto  --seed 33 --iterations 1000  --append-expr --load_domain dumps/abl-unlock33-load=scratch.pth
```

### Evaluation

```bash
jac-run babyai_src/babyai-abil-bc.py minigrid goto  --seed 33 --iterations 1000  --append-expr --load_domain dumps/abl-unlock33-load=scratch.pth --load dumps/seed33/abil-bc-goto-load=scratch.pth --evaluate
```


## BibTeX

```
@misc{shao2024learninglonghorizonplanningneurosymbolic,
      title={Learning for Long-Horizon Planning via Neuro-Symbolic Abductive Imitation}, 
      author={Jie-Jing Shao and Hao-Ran Hao and Xiao-Wen Yang and Yu-Feng Li},
      year={2024},
      eprint={2411.18201},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.18201}, 
}
```

## Acknowledgement
[PDSketch](https://github.com/vacancy/PDSketch-Alpha-Release), [BabyAI](https://github.com/mila-iqia/babyai), [Mini-BEHAVIOR](https://github.com/StanfordVL/mini_behavior), [CLIPort](https://github.com/cliport/cliport)