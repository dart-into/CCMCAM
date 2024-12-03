# Bidirectional Feature Fusion via Cross-Attention Transformer for Chrysanthemum Classification
![swinca4](https://github.com/user-attachments/assets/84281dbf-6471-48fc-bb2b-9c7b66c376a1)
## File Introduction
### main.py
main.py is the main entry point of the program. It contains the core logic for executing the classification task.
### Mutiscale.py
Mutiscale.py contains the network architecture code. It defines the model structure and is essential for the classification process.
### loop.py
loop.py is used to repeatedly execute main.py multiple times to average the test results, ensuring more reliable performance metrics.
### Mydataset.py
Mydataset.py inherits from the dataset class and is responsible for loading and processing the dataset used in the classification task.
## package
### The required libraries are as follows (some of which may not be useful, for reference only):
### Package   Version
### einops==0.8.0
### matplotlib==3.4.3
### net==2.4
### numpy==1.19.5
### Pillow==8.3.2
### Pillow==11.0.0
### scikit_learn==0.23.2
### timm==1.0.11
### torch==1.10.0+cu111
### torchvision==0.11.1+cu111
### tqdm==4.61.2
