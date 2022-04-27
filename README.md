## In-BoXBART ##

An instruction-based unified model for performing various biomedical tasks.

You may want to check out 
* Our paper (NAACL 2022 Findings): [In-BoXBART: Get Instructions into Biomedical Multi-Task Learning](https://arxiv.org/abs/2204.07600)
* GitHub: [Click Here](https://github.com/Mihir3009/In-BoXBART)

This work explores the impact of instructional prompts on biomedical Multi-Task Learning. We introduce the BoX, a collection of 32 instruction tasks for Biomedical NLP across (X) various categories. Using this meta-dataset, we propose a unified model termed In-BoXBART, that can jointly learn all tasks of the BoX without any task-specific modules. To the best of our knowledge, this is the first attempt to
propose a unified model in the biomedical domain and use instructions to achieve generalization across several biomedical tasks.

## How to Use ##

You can very easily load the models with Transformers, instead of downloading them manually. The BART-base model is the backbone of our model. Here is how to use the model in PyTorch:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("cogint/in-boxbart")
model = AutoModelForSeq2SeqLM.from_pretrained("cogint/in-boxbart")
```
Or just clone the model repo
```
git lfs install
git clone https://huggingface.co/cogint/in-boxbart
```

## BibTeX Entry and Citation Info ##

If you are using our model, please cite our paper:

```bibtex
@article{parmar2022boxbart,
  title={{In-BoXBART: Get Instructions into Biomedical Multi-Task Learning}},
  author={Parmar, Mihir and Mishra, Swaroop and Purohit, Mirali and Luo, Man and Murad, M Hassan and Baral, Chitta},
  journal={NAACL 2022 Findings},
  year={2022}
}
```
