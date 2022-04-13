#BERT Implementation
This is a "bare essentials" implementation of the orignal BERT paper:
https://arxiv.org/pdf/1810.04805.pdf.

The purpose of this project is practice and understanding BERT.

The model is implemented in PyTorch. It uses the original Google checkpoint file for 
loading the pretrained model. The checkpoint used is uncased_L-12_H-768_A-12. 
It can be found and downloaded the original BERT github: 
https://github.com/google-research/bert.

With the pretrained model loaded, this implementation finetunes the specified task and 
achieves evaluation results as described in the paper.

Most of the preprocessing functions were taken directly from the original BERT code. 
For the SQuAD task, they used some heuristics for better predicitions, so I've included
these as well.

### Sentence (and sentence-pair) classification tasks