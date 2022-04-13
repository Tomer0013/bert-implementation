#BERT Implementation
This is a "bare essentials" implementation of the orignal BERT paper:
https://arxiv.org/pdf/1810.04805.pdf.

The model is implemented in PyTorch. It uses the original Google checkpoint file for the 
bert_base_pretrained model. The checkpoint can be found and downloaded from their github: 
https://github.com/google-research/bert.

With the pretrained model loaded, this implementation finetunes the specified task and 
achieves evaluation results as described in the paper.