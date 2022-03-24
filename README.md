# Cassava Leaf Disease Classification - Pytorch project
Leaf disease image classification model deployed with Streamlit. Project contains also implemented from scratch [MLP-Mixer](https://arxiv.org/abs/2105.01601v3) [1] architecture.

### Content
- `main.py` - Streamlit server code
- `eval.py` - supplementary functions for model inference in Streamlit
- `models.py` - model definitions and Pytorch Lightning model wrapper
- `train.py` - training script with early stopping and tensorboard logging.



### Data
Dataset comes from [Cassava Leaf Disease Kaggle competition](https://www.kaggle.com/c/cassava-disease/data)[2] as .png files labeled with `.csv` file.

### Models
`MlpMixer` - implemented neural network as described in paper[1].

`TransferredInception` - class for transfering Inception_v3 architecture.

`LightningModelWrapper` - Pytorch Lightning wrapper for models above (or any other for this task). Comes with logging per epoch (for better more readable learnig process plots in Tensorboard)

### Deployment
Application is served on Microsoft Azure VM with Streamlit on address: 
http://20.91.220.251

### Used tools
- Pytorch
- Pytorch Lightning
- Albumentations
- Streamlit

## References
[1]I. Tolstikhin et al., ‘MLP-Mixer: An all-MLP Architecture for Vision’, arXiv:2105.01601 [cs], Jun. 2021, Accessed: Mar. 14, 2022. [Online]. Available: http://arxiv.org/abs/2105.01601

[2]E. Mwebaze, T. Gebru, A. Frome, S. Nsumba, and J. Tusubira, ‘iCassava 2019 Fine-Grained Visual Categorization Challenge’, arXiv:1908.02900 [cs], Dec. 2019, Accessed: Mar. 14, 2022. [Online]. Available: http://arxiv.org/abs/1908.02900



