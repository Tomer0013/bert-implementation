# BERT Implementation
This is a "bare essentials" implementation of the orignal BERT paper:
https://arxiv.org/pdf/1810.04805.pdf.

The purpose of this project is practice and understanding BERT.

The model is implemented in PyTorch. It uses the original Google checkpoint file for 
loading the pretrained model. The checkpoint used is `uncased_L-12_H-768_A-12`. 
It can be found and downloaded the original BERT github: 
https://github.com/google-research/bert.
The scripts expects the checkpoint files to be located within 
`bert_base_pretrained/uncased_L-12_H-768_A-12`.

With the pretrained model loaded, this implementation finetunes the specified task and 
achieves evaluation results as described in the paper (for GlUE tasks, evaluation is
performed on the dev test. The results are very similar to those mentioned in the paper).

Most of the preprocessing functions were taken directly from the original BERT code. 
For the SQuAD task, they had used some heuristics for better predicitions, so these 
were included as well.

### Instructions
The fine-tuning tasks implemented are the GLUE classification tasks and the question 
answering on the SQuAD dataset. 

For the GLUE tasks, the script expects `train.tsv` and `dev.tsv` to be located in each of the
subfolders within `datasets/glue_data`. These can be downloaded following the instructions on the
GLUE website.

For the SQuAD task, the script expects `train-v2.0.json` and `dev-v2.0.json` to be
located in the `datasets/squad_data` folder. It can be downloaded from the SQuAD website.

Following are the execution commands for each of the tasks, and the expected output. The
evaluation metrics chosen are the same as in the paper. For the classification tasks I've
also included the accuracy. 

The script uses the parameter values described in the paper 
and in the original code as defaults, therefore only the task name is required. 
Please check `args.py` if you wish to execute it with different parameters.

All of the tasks ran fine on an Ubuntu based machine with a GeForce RTX 3080 16MB and
32MB RAM.

### GLUE Classification Tasks
#### RTE
```
python run_glue_finetune.py --task_name rte
```
```
***** Eval results *****
eval_accuracy: 0.714801
```
#### MRPC
```
python run_glue_finetune.py --task_name mrpc --lr 2e-5
```
```
***** Eval results *****
eval_accuracy: 0.860294
eval_f1_score: 0.901213
```
#### STS-B
```
python run_glue_finetune.py --task_name sts-b
```
```
***** Eval results *****
eval_spearman_corr: 0.893098
```
#### CoLA
```
python run_glue_finetune.py --task_name cola
```
```
***** Eval results *****
eval_accuracy: 0.833174
eval_matthews_corr: 0.590959
```
#### SST-2
```
python run_glue_finetune.py --task_name sst-2
```
```
***** Eval results *****
eval_accuracy: 0.923165
eval_loss: 0.270706
```
#### QNLI
```
python run_glue_finetune.py --task_name qnli
```
```
***** Eval results *****
eval_accuracy: 0.912868
eval_loss: 0.370207
```
#### QQP
```
python run_glue_finetune.py --task_name qqp
```
```
***** Eval results *****

```
#### MNLI
```
python run_glue_finetune.py --task_name mnli
```
```
***** Eval results *****

```