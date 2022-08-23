## In-BoXBART ##

An instruction-based unified model for performing various biomedical tasks.

You may want to check out 
* Our paper (NAACL 2022 Findings): [In-BoXBART: Get Instructions into Biomedical Multi-Task Learning](https://arxiv.org/abs/2204.07600)
* Hugging Face: [cogint/in-boxbart](https://huggingface.co/cogint/in-boxbart)

This work explores the impact of instructional prompts on biomedical Multi-Task Learning. We introduce the BoX, a collection of 32 instruction tasks for Biomedical NLP across (X) various categories. Using this meta-dataset, we propose a unified model termed In-BoXBART, that can jointly learn all tasks of the BoX without any task-specific modules. To the best of our knowledge, this is the first attempt to propose a unified model in the biomedical domain and use instructions to achieve generalization across several biomedical tasks. Below figure shows the overview of the approach.

<p align="center"><img src="https://user-images.githubusercontent.com/47143544/166615962-4e533d10-e017-4439-9312-976f69b0377f.png" width="450"></p>

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

## Biomedical Instructions ##

Please see `./templates` to find instructional prompts corresponsing to all tasks from BoX that are used for experiments. For more details, refer [our paper](https://arxiv.org/abs/2204.07600).

## Quick start ##

Find below details to use our source code for fine-tuning other models.

> #### Setup

Run the following command to install all the dependecies to run the model:
```python
pip install -r requirements.txt
```

> #### Training

In order to finetune model on your data, use `scripts/finetune_model.py` and run it with the following arguments:

```
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

To evaluate model, use `scripts/evaluation.py` and run it with the following arguments:

```
python evaluation.py \

            --dataset_file              Path of test data file (a JSON or CSV file), which contains ground truth.
                                        (default: None) \
            --prediction_file           Path of the prediction file; expected to be a JSON file of the following format: { "predictions": ["pred1", "pred2", ...] } or .txt file of the following format:  "pred1" \n "pred2" \n ...
                                        (default: None) \
            --save_results              The output directory where JSON of result will be saved.
                                        (default: None)
```

## Inference Example ##

Here, we provide an example for the "Document Classification" (HoC dataset) task. Once you load the model from huggigface for inference, you can append instruction given in `./templates` for that particular dataset with input instance. Below is an example of one instance.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("cogint/in-boxbart")
model = AutoModelForSeq2SeqLM.from_pretrained("cogint/in-boxbart")

# Input shows how we have appended instruction from our file for HoC dataset with instance.

input = "Instruction: Definition: In this task, you are given a medical text related to cancer. Your job is to classify into zero or more classes from (1) Sustaining proliferative signaling, (2) Resisting cell death, (3) Genomic instability and mutation, (4) Activating invasion and metastasis, (5) Tumor promoting inflammation, (6) Evading growth suppressors, (7) Inducing angiogenesis (8) Enabling replicative immortality, (9) Avoiding immune destruction and (10) Cellular energetics., Positive Examples: [[input: Studies of cell-cycle progression showed that the anti-proliferative effect of Fan was associated with an increase in the G1/S phase of PC3 cells ., output: Evading growth suppressors, Sustaining proliferative signaling, explanation: Given text is classified into two categories, hence, generated label is 'Evading growth suppressors, Sustaining proliferative signaling'.] ]; Instance: input: Similar to previous studies utilizing IGF-1 , pretreatment with Roscovitine leads to a significant up-regulation of p21 expression and a significant decrease in the number of PCNA positive cells ., output: ?"

# Ideal output for this input is 'Sustaining proliferative signaling'

output = model(input)
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
* For help or issues in using In-BoXBART, please submit a GitHub issue.
* Please contact Mihir Parmar (mparmar3@asu.edu) or Mirali Purohit (mpurohi3@asu.edu) for communication related to In-BoXBART.
