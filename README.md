# Bi-ACTCNet
Bi-ACTCNet: A Bidirectional Channel Attention and Mutual-Cross-Attention Temporal Feature Extraction Network for Motor-Rest Classification in Motor Imagery

Author: Lei Cao, Xiangrui Cai*, Yiling Dong.

# About Bi-ACTCNet
1. We focus for the first time on the Motor-Rest Classification task and construct the Bi-ACTCNet model to effectively decode this type of task, thus filling a research gap in this area.
   
2. We validate the effectiveness of the model on three Motor-Rest Classification datasets.
   
3. We propose a bidirectional temporal feature extraction signal processing method and fuse the features of two dimensions using MCA, demonstrating that the reverse features in EEG signals contain additional information that helps improve model accuracy.
   
4. We address the issue of insufficient channel interactivity in the ECA channel attention block by proposing the ETCA block.

5. To verify the effectiveness and generalization of the model, we tested it on the 2-class classification of the BCI Competition IV 2a dataset, achieving good results.
   
![Model](./Model.png)

# About Datasets
We used four datasets: the Motor Attempt or Resting State Dataset, BCI Competition IV Dataset 1, Physionet Dataset, and BCI Competition IV 2a Dataset. Among them, the Motor Attempt or Resting State Dataset is a private dataset, while the other three are publicly available datasets that can be downloaded and used as input to the model after preprocessing.

# Development environment
We implemented our Bi-ACTCNet model using Python 3.8 and the TensorFlow library on a NVIDIA GeForce RTX 4060 Laptop GPU.

· Tensorflow == 2.9

· matplotlib == 3.7.5

· NumPy == 1.23.5

· scikit-learn == 1.1.3

· SciPy == 1.9.3

# References
The article has currently been accepted by the IEEE Internet of Things Journal.

If you have any questions about our paper or code, please feel free to contact us at: 202330310112@stu.shmtu.edu.cn

# Other
We would like to express our sincere gratitude to the following papers for providing open-source code, which has been of great significance to our research:

Physics-informed attention temporal convolutional network for EEG-based motor imagery classification.

Feature Fusion Based on Mutual-Cross-Attention Mechanism for EEG Emotion Recognition.

# Details
The data loading script, training code, and model weights for the BCI Competition IV Dataset 1 are provided in preprocess_bci41.py, main_bci41.py, and results_41_fold12, respectively. Readers who are interested can reproduce the results using the model weights included in the results_41_fold12 file.

