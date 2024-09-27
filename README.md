# GCLink: A Graph Contrastive Link Prediction Framework for Gene Regulatory Network Inference

## Dependencies Used in This Project
- Python 3.8.0
- PyTorch 1.12.1
- torch-scatter 2.1.0
- torch-sparse 0.6.15
- torch-geometric 2.3.1

## Usage
1. **Gene Regulatory Network Inference**:
   - First, run the `Train_Test_Split.py` function to generate training, validation, and test sets.
   - Then, execute `GCLink_main.py` for inference.

2. **Transfer Learning**:
   - First, run `train_source.py` to train the source model.
   - Then, execute `transfer.py` for transfer learning.

## Data
The source data format is illustrated in the "Specific Dataset" folder. The example data in the "Data" folder shows the results obtained after running `Train_Test_Split.py`.

## Contact
For any inquiries, feel free to raise issues or contact me via email at yoyiming7@gmail.com.
