# Multi-Model Sentiment Classification Pipeline  
### Controlled Ablation Study with Hugging Face Transformers

A complete, reproducible pipeline for **fine-tuning** Transformer models on a 3-class sentiment classification task.  
All experiments are designed with a fixed seed, locked data partitions, and systematic logging to ensure transparent and comparable results.

---

## Key Features

- Supports 5 different models:
  - `bert-base-cased`
  - `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
  - `albert/albert-base-v2`
  - `vinai/bertweet-base`

- Four independent experimental arms (Ablation Studies):
  1. Learning Rate × Optimizer + Epoch Sweep  
  2. Preprocessing (raw vs cleaned)  
  3. Max Sequence Length (64 and 128)  
  4. Fine-tuning Strategy (full vs frozen encoder)

- Mixed Precision Training (AMP) on GPU
- Full reproducibility control (seed = 42)
- Tracks the best configuration per model based on Dev F1 and stores test errors
- Produces 7 standardized CSV reports ready for analysis

---

## Data Structure

The script expects two text files in the same directory:
train_text.txt
train_labels.txt

Data partitioning is performed exactly as follows (no shuffling):

| Split     | Index Range       | Number of Samples |
|-----------|-------------------|-------------------|
| Train     | 0 : 10000         | 10,000            |
| Dev       | 10000 : 12000     | 2,000             |
| Test      | 12000 : 20000     | 8,000             |

---

## Base Configuration (CONFIG)

```python
CONFIG = {
    "batch_size": 16,
    "seed_value": 42,
    "default_lr": 2e-5,
    "default_epochs": 2,
    "default_len": 96,
    "device": "cuda" if available else "cpu"
}

```
---
## Experimental Arms
Arm 1 – Learning Rate & Optimizer
Learning rates: 2e-5 and 5e-5
Optimizers: AdamW and SGD (momentum=0.9)
For the default combination (lr=2e-5 + AdamW), training runs up to 4 epochs (Epoch Sweep)
Arm 2 – Preprocessing Ablation
raw: original text without modification
cleaned: lowercase + remove URLs and @mentions + normalize whitespace
Arm 3 – Sequence Length Ablation
Tested lengths: 64 and 128
(Default length of 96 is used in the other arms)
Arm 4 – Fine-tuning Strategy
full: all parameters are trainable
frozen_encoder: only the classifier layer (and pooler/pre_classifier if present) is trained
In all arms, the winning model is selected based on the highest Dev F1, and its test metrics + misclassified examples are saved.

## Outputs
After a full run, the script generates the following CSV files:

|                File	            |                    Content                     |
|-----------------------------------|------------------------------------------------|
|1_hyperparameters_report.csv       |	Results of all LR × Optimizer combinations   |
|1b_epoch_sweep_report.csv	        | Epoch sweep results (2 and 4 epochs)           |
|2_preprocessing_report.csv	        |Comparison of raw vs cleaned                    |
|3_sequence_length_report.csv	    |Comparison of lengths 64 and 128                |
|4b_finetune_strategy_report.csv	|Comparison of full vs frozen_encoder            |
|4_final_models_benchmark.csv	    |Best configuration + test metrics for each model|
|5_unified_error_analysis.csv	    |All misclassified test samples                  |
|6_fixed_baseline_comparison.csv	|All models under identical baseline conditions  |

## How to Run
Requirements
```bash
pip install torch transformers scikit-learn pandas numpy
```
Execution
```bash
python your_script_name.py
```
The script will automatically:

Detect GPU availability
Process all models sequentially
Clear memory after each experiment
Save the seven report files

## Important Notes
Every experiment starts with enforce_reproducibility(42).
ignore_mismatched_sizes=True is used so pre-trained models with different number of classes can be loaded easily.
BERTweet uses normalization=True in the tokenizer.
Fine-tuning strategies and preprocessing methods are tested independently (not a full factorial design).

## License & Usage
This code is written for research and educational purposes.
Feel free to use, modify, and publish results based on it.

Built for transparency, reproducibility, and fair model comparison.