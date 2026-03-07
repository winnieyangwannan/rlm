# Task: tweet-sentiment-extraction | Rollout 3637 : Percentile: 0.29 | Tier: 🔴 Low | Medal: None

---

## Summary

The solution frames tweet sentiment extraction as extractive question answering by pairing the sentiment string as the question and the tweet as the context. It fine-tunes a Hugging Face AutoModelForQuestionAnswering (default bert-base-uncased) on start/end token positions derived from selected_text, using robust char-span matching and token offset mappings. At inference, it masks to context tokens, selects the best span maximizing start+end logits, and for 'neutral' tweets returns the full text.

---

## Task Analysis

goal: Extract the substring (selected_text) from each tweet that best supports the provided sentiment label.

task_type: extractive span selection (start/end indices within text)

data_modality: text

evaluation_metric: Average word-level Jaccard similarity between ground-truth selected_text and predictions (higher is better), computed by splitting on whitespace and lowercasing.

core_challenges: Precisely aligning character spans in noisy tweet text to token boundaries; handling casing and whitespace variation; ambiguous 'neutral' cases where full text may be appropriate.

difficulty_factors: Noisy punctuation/casing and inconsistent spacing in tweets; multiple plausible substrings can yield similar Jaccard; exact substring extraction requires robust char-to-token mapping.

---

## Data Preprocessing

cleaning: Casts text/sentiment/selected_text to string; maps selected_text to character spans with a robust finder (case-insensitive search, trimming, space-collapsing heuristic); falls back to full text if not found.

transformations: Tokenization with AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) on pairs (sentiment, text), with return_offsets_mapping=True; sequence_ids used to isolate text tokens (sequence_id == 1); converts char spans to start/end token indices.

pipeline_structure: Tokenizer loaded from pretrained and applied separately to train and test; training drops offset_mapping from encodings and supplies start_positions/end_positions; inference reuses test encodings with offset_mapping and sequence_ids for span decoding.

augmentation: null

---

## Feature Engineering

new_features: none

selection_or_reduction: none

---

## Model Selection

algorithms: transformers.AutoModelForQuestionAnswering (default backbone 'bert-base-uncased') trained via transformers.Trainer

hyperparameters: TrainingArguments(num_train_epochs=2 (or 1 if FAST_DEV), per_device_train_batch_size=16, per_device_eval_batch_size=16, learning_rate=2e-5, weight_decay=0.01, logging_steps=100, save_steps=0, evaluation_strategy='no', fp16=CUDA-available, dataloader_num_workers=2, seed=SEED, report_to=[])

ensemble: null

pretrained_models: Hugging Face pretrained QA head initialized from MODEL_NAME (default 'bert-base-uncased'), fine-tuned end-to-end

---

## Training Methodology

objective_alignment: Trains the QA model with cross-entropy over start/end token indices; this does not directly optimize the word-level Jaccard metric.

validation_strategy: No validation: evaluation_strategy='no', no holdout or cross-validation, no early stopping.

training_configuration: Trainer with DataCollatorWithPadding; epochs=2 (or 1 with FAST_DEV), batch size=16, learning_rate=2e-5, weight_decay=0.01, mixed precision enabled if CUDA; seeds set for Python, NumPy, and Torch.

tuning: None

---

## Evaluation and Submission

prediction_method: For each test example, restrict logits to context tokens (sequence_id == 1), slice to actual seq_len via attention_mask, and select the span maximizing start_logits[i] + end_logits[j] with j >= i; map token offsets back to original text; for 'neutral' sentiment return full text.

post_processing: If the decoded span is invalid (empty or bad offsets), fall back to returning the original full text; no additional text normalization or calibration.

---

## Notable Implementation Details

Includes a robust find_char_span with case-insensitive search, trimming, and space-collapsing plus head/tail remapping; uses Hugging Face fast tokenizer offset_mapping and sequence_ids to align spans; masks non-context tokens by setting logits to -1e9; FAST_DEV mode samples up to 2000 train rows and uses 1 epoch; installs transformers==4.39.3/torch if missing; runs mixed precision on GPU; batched inference with DataCollatorWithPadding; submission saved to /workspace/submission.csv.