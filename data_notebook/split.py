import logging
import sys
import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# --- Import core components from your project files ---
# NOTE: Assuming ModelArguments, DataTrainingArguments, and OurTrainingArguments 
# are defined in the file 'train.py'.
try:
    from trainer import SelfMixTrainer
    from model import Bert4Classify
    from datasets import load_dataset, SelfMixData
    from train import ModelArguments, DataTrainingArguments, OurTrainingArguments 
    from transformers import AutoTokenizer, HfArgumentParser, set_seed
except ImportError as e:
    logging.error(f"Failed to import core classes. Ensure files (trainer.py, model.py, datasets.py, train.py) are accessible and contain required classes: {e}")
    sys.exit(1)


logger = logging.getLogger(__name__)

# --- Main Execution ---

def main():
    # 1. Argument Parsing and Setup
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    
    # Handle JSON config and command-line arguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 1:
        # Use default arguments if no command line arguments or JSON file is provided
        model_args, data_args, training_args = ModelArguments(), DataTrainingArguments(), OurTrainingArguments()
    else:
        # Standard command line argument parsing
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    set_seed(training_args.seed) 
    
    # Check for file existence
    if not os.path.exists(data_args.train_file_path):
        logger.error(f"Training data file not found at: {data_args.train_file_path}. Please provide a valid CSV file.")
        sys.exit(1)
        
    if not os.path.exists(data_args.eval_file_path):
        logger.warning(f"Evaluation data file not found at: {data_args.eval_file_path}. Continuing...")

    try:
        # 2. Data Loading and Preparation
        logger.info("--- Loading data and model components ---")
        
        # Load data (This is the original, full dataset)
        train_datasets, train_num_classes = load_dataset(data_args.train_file_path, data_args.dataset_name)
        
        # Get the original texts and true labels (if available) for output
        original_texts = train_datasets.texts
        original_labels = train_datasets.labels
        
        eval_datasets, _ = load_dataset(data_args.eval_file_path, data_args.dataset_name)
        model_args.num_classes = train_num_classes
        
        # Tokenizer and Data Wrapper
        tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
        tokenizer.add_tokens(["{", "}", "<span>", "</span>","N/A"]) 
        selfmix_train_data = SelfMixData(data_args, train_datasets, tokenizer)
        selfmix_eval_data = SelfMixData(data_args, eval_datasets, tokenizer)
        
        # 3. Load Model and Initialize Trainer
        model = Bert4Classify(model_args.pretrained_model_name_or_path, model_args.dropout_rate, model_args.num_classes)
        
        if model_args.checkpoint_path is not None:
            model.load_model(model_args.checkpoint_path)
            print(f"Loaded model from {model_args.checkpoint_path}")
        else:
            raise ValueError("Please provide a valid checkpoint_path to load the model for splitting.")
        
        # NOTE: We initialize the trainer even though we won't call trainer.train(), 
        # as we rely on its internal methods for the GMM split logic.
        trainer = SelfMixTrainer(
            model=model,
            train_data=selfmix_train_data,
            eval_data=selfmix_eval_data,
            model_args=model_args,
            training_args=training_args
        )

        # 4. Execute the Data Split Core Logic to get the mask and probabilities
        logger.info("\n--- EXECUTING DYNAMIC DATA SPLIT CORE LOGIC ---")
        
        # The trainer needs the 'all' loader to calculate losses on all samples
        train_loader_all = trainer.train_data.run("all")
        
        # Step 1: Get probabilities from GMM fit (via _eval_samples, which handles NaNs)
        prob = trainer._eval_samples(train_loader_all) 
        
        # Step 2: Get the boolean prediction mask (True=Labeled, False=Unlabeled)
        pred_mask = (prob > trainer.model_args.p_threshold)
        
        # 5. CREATE AND SAVE THE FINAL CSV
        
        # Convert boolean mask to descriptive tags
        # tags = np.where(pred_mask, 'labeled', 'unlabeled')
        tags = np.where(pred_mask, 0, 1)
        
        # Create a DataFrame for output
        output_df = pd.DataFrame({
            'tag': original_labels,
            'text': original_texts,
            'noise': tags,
            'clean_probability': prob 
        })
        
        output_filename = 'data_split_output.csv'
        output_df.to_csv(output_filename, index=False)
        logger.info(f"\n*** SUCCESS: Split data saved to '{output_filename}' ***")
        logger.info(f"Labeled (Clean) Samples: {pred_mask.sum()}")
        logger.info(f"Unlabeled (Noisy) Samples: {len(original_texts) - pred_mask.sum()}")
        
        # Optionally inspect the split loaders (as per previous requests)
        # Note: We call get_labeled_unlabeled_loaders *after* calculating prob and pred_mask 
        # to ensure the CSV output is based on the logic we just executed.
        labeled_loader, unlabeled_loader = trainer.get_labeled_unlabeled_loaders()
        
        if len(labeled_loader.dataset) > 0:
             input_ids, att_mask, label, prob, index = next(iter(labeled_loader))
             logger.info(f"First Labeled Batch Sample Index: {index[0].item()}, Clean Probability: {prob[0].item():.4f}, len: {len(labeled_loader.dataset)}")
        
        if len(unlabeled_loader.dataset) > 0:
             input_ids, att_mask, index = next(iter(unlabeled_loader))
             logger.info(f"First Unlabeled Batch Sample Index: {index[0].item()} ,len: {len(unlabeled_loader.dataset)}")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        # Optionally, re-raise the exception for debugging
        # raise

if __name__ == "__main__":
    main()
