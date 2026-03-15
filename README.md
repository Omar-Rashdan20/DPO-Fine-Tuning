# DPO-Fine-Tuning

This notebook demonstrates **Direct Preference Optimization (DPO)** fine-tuning of a **LLaMA-3 language model** using **QLoRA, TRL, and the UltraFeedback dataset**, designed to run efficiently on a **single Kaggle NVIDIA T4 GPU**.

The project provides a **complete end-to-end alignment pipeline**, showing how to train a preference-aligned LLM without a reward model while keeping GPU memory usage low through **4-bit quantization and LoRA adapters**.

## What the notebook covers

The workflow walks through the entire DPO training process:

1. Environment setup and dependency installation  
2. Loading the UltraFeedback dataset (`BarraHome/ultrafeedback_binarized`)  
3. Configuring **4-bit quantization** for memory-efficient training  
4. Loading the **LLaMA-3 base model and tokenizer**  
5. Preparing the model for **k-bit training**  
6. Applying **LoRA adapters** to attention layers  
7. Loading a **frozen reference model** for stable DPO optimization  
8. Training using **TRL's `DPOTrainer`** with preference pairs  
9. Saving the trained LoRA adapter  
10. Running **inference comparisons** between the base and aligned models  
11. **Merging LoRA weights into the base model** for deployment

## Dataset

The training data comes from:

`BarraHome/ultrafeedback_binarized`

This dataset is a cleaned version of **UltraFeedback** containing **human preference annotations**. Each record includes:

- a prompt `x`
- a **chosen response** `y_w`
- a **rejected response** `y_l`

To keep the experiment lightweight and reproducible on Kaggle:

- **500 samples** are used for training  
- **30 samples** are used for evaluation

This subset is sufficient to demonstrate **DPO training dynamics** within the constraints of a single T4 GPU.

## Key Techniques Demonstrated

- **Direct Preference Optimization (DPO)** for LLM alignment
- **QLoRA (4-bit quantization + LoRA)** for efficient fine-tuning
- **TRL's `DPOTrainer`** for preference-based training
- **Explicit frozen reference model** to stabilize optimization
- **LoRA weight merging** for production-ready models

## Final Output

The notebook produces a **merged model without PEFT dependencies**, which can be:

- deployed directly
- further quantized
- exported to formats such as **GGUF** for inference frameworks like `llama.cpp`
