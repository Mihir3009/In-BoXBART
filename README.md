## In-BoXBART ##

An instruction-based unified model for performing various biomedical tasks.

You may want to check out 
* Our paper (NAACL 2022 Findings): [In-BoXBART: Get Instructions into Biomedical Multi-Task Learning](https://arxiv.org/abs/2204.07600)
* Hugging Face: [cogint/in-boxbart](https://huggingface.co/cogint/in-boxbart)

This work explores the impact of instructional prompts on biomedical Multi-Task Learning. We introduce the BoX, a collection of 32 instruction tasks for Biomedical NLP across (X) various categories. Using this meta-dataset, we propose a unified model termed In-BoXBART, that can jointly learn all tasks of the BoX without any task-specific modules. To the best of our knowledge, t(his is the first attempt to propose a unified model in the biomedical domain and use instructions to achieve generalization across several biomedical tasks.)

<!--![In-BoXBART](https://drive.google.com/file/d/1YbhO0a9bPbOJEZKulrqhkoUZUduAlTFH/view?usp=sharing)-->

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

## Quick start ##

> #### Setup

Run the following command to install all the dependecies to run the model:
```python
pip install -r requirements.txt
```

> #### Training

In order to finetune model on your data, use [scripts/finetune_model.py](https://github.com/Mirzyaaliii/In-BoXBART/blob/main/scripts/finetune_model.py) and run it with the following arguments:

```python
python scripts/finetune_model.py \

        --model_name_or_path            Provide path of the model, you want to finetune. To finetune on BART use - "facebook/bart-base"
                                        (default: None) \
        --do_train                      Provide True or False
                                        (default: False) \
        --do_eval                       Provide True or False
                                        (default: False) \
        --do_predict                    Provide True or False
                                        (default: False) \
        --train_file                    Path of an optional input training data file (a JSON or CSV file), if do_train argument is true.
                                        (default: None) \
        --validation_file               Path of an optional input evaluation data file to evaluate the metrics (rouge) on (a JSON or CSV file), if do_eval argument is true.
                                        (default: None) \
        --test_file                     Path of an optional input test data file to evaluate the metrics (rouge) on (a JSON or CSV file), if do_predict argument is true.
                                        (default: None) \
        --output_dir                    The output directory where the model predictions and checkpoints will be written.
                                        (default: None) \
        --per_device_train_batch_size   Batch size per GPU/TPU core/CPU for training.
                                        (default: 8) \
        --per_device_eval_batch_size    Batch size per GPU/TPU core/CPU for evaluation.
                                        (default: 8) \
        --gradient_accumulation_steps   Number of updates steps to accumulate before performing a backward/update pass.
                                        (default: 1) \
        --predict_with_generate         Whether to use generate to calculate generative metrics (ROUGE, BLEU).
                                        (default: False) \
        --save_strategy                 The checkpoint save strategy to use. (no, steps, epoch)
                                        (default: steps)
```


> #### Evaluation

To evaluate model, use [scripts/evaluation.py](https://github.com/Mirzyaaliii/In-BoXBART/blob/main/scripts/evaluation.py) and run it with the following arguments:

```python
python evaluation.py \

            --dataset_file              Path of test data file (a JSON or CSV file), which contains ground truth.
                                        (default: None) \
            --prediction_file           Path of the prediction file; expected to be a JSON file of the following format: { "predictions": ["pred1", "pred2", ...] } or .txt file of the following format:  "pred1" \n "pred2" \n ...
                                        (default: None) \
            --save_results              The output directory where JSON of result will be saved.
                                        (default: None)
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

## Contact Information ##
* For help or issues using In-BoXBART, please submit a GitHub issue.
* Please contact Mihir Parmar (mparmar3@asu.edu) or Mirali Purohit (mpurohi3@asu.edu) for communication related to In-BoXBART.
