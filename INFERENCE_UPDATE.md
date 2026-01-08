# Test Inference & Evaluation Update

## Summary

Updated test inference workflow to use Unsloth FastVisionModel architecture, matching the fine-tuning script. The new implementation provides better compatibility, proper tokenization, and automatic evaluation report generation.

## Changes Made

### 1. New Files Created

- **`utils/test_inference_unsloth.py`** - New test inference script using Unsloth FastVisionModel
  - Proper tokenization with `apply_chat_template()`
  - Extracts only new tokens from generation
  - Handles video paths with subdirectories (extracts filename only)
  - Better generation parameters: `temperature=0.1`, `do_sample=True`
  - Default data path: `videos/` (flat directory structure)

### 2. Updated Files

- **`scripts/run_inference.sh`**
  - Uses `test_inference_unsloth.py` instead of old `test_inference.py`
  - Improved error handling (continues to report generation even if inference has errors)
  - Default model: `unsloth/gemma-3n-E4B-it`
  - Default data path: `videos/`
  - Added `--num_frames` parameter (default: 8)
  - Removed `--base_model` parameter (not needed with Unsloth)

- **`utils/generate_test_report.py`**
  - Added `import os` (was missing)
  - Better error handling with proper exit codes
  - Validates predictions file exists before processing
  - Shows helpful error messages

- **Documentation Updates:**
  - `README.md` - Updated test inference section with new workflow
  - `utils/README.md` - Updated script references and examples
  - `docs/QUICKSTART.md` - Updated evaluation commands
  - `docs/SETUP_CHECKLIST.md` - Updated inference examples
  - `gemma3/docs/QUICKSTART.md` - Updated evaluation section

### 3. Key Improvements

**Tokenization Fix:**
- Old: Used separate `tokenizer()` call with frames, causing output issues
- New: Uses `tokenizer.apply_chat_template()` with proper parameters

**Video Path Handling:**
- Old: Expected videos in subdirectories matching JSON paths
- New: Extracts filename only, works with flat `videos/` directory

**Generation Parameters:**
- Old: `temperature=1.5, min_p=0.1` (unstable)
- New: `temperature=0.1, do_sample=True` (more consistent)

**Output Decoding:**
- Old: Decoded full sequence, then split by "assistant\n\n"
- New: Decodes only new tokens, cleaner output

## Usage

### Quick Start

```bash
# Run test inference with fine-tuned model (50 samples by default)
bash scripts/run_inference.sh \
    --model_path outputs/gemma3n_finetune_20260108_162806_merged_16bit

# Test with base model for comparison
bash scripts/run_inference.sh \
    --model_path unsloth/gemma-3n-E4B-it

# Full test set (no limit)
bash scripts/run_inference.sh \
    --model_path outputs/gemma3n_finetune_merged_16bit \
    --limit ""

# Fast evaluation (skip BERT, only ROUGE/METEOR)
bash scripts/run_inference.sh \
    --model_path outputs/gemma3n_finetune_merged_16bit \
    --no-bert
```

### Direct Script Usage

```bash
# Run inference only (without report generation)
python utils/test_inference_unsloth.py \
    --model_path outputs/gemma3n_finetune_merged_16bit \
    --test_json dataset/qved_test.json \
    --data_path videos \
    --limit 10

# Generate report from existing predictions
python utils/generate_test_report.py \
    --predictions results/test_inference_model/test_predictions.json \
    --output test_report.xlsx
```

## Output Structure

```
results/test_inference_<model_name>/
├── test_predictions.json          # Predictions with ground truth
└── test_evaluation_report.xlsx    # Excel report with:
    ├── Detailed Results sheet (BERT/ROUGE/METEOR per sample)
    ├── Summary Statistics
    ├── Exercise Accuracy
    └── Charts
```

## Requirements

All requirements already in `requirements.txt`:
- `unsloth` - FastVisionModel
- `evaluate` - Evaluation metrics
- `nltk` - BLEU scores
- `rouge-score` - ROUGE metrics
- `sentence-transformers` - BERT embeddings
- `openpyxl` - Excel report generation
- `scikit-learn` - Cosine similarity

## Migration Notes

**If you were using the old `test_inference.py`:**

1. Replace with `test_inference_unsloth.py`
2. Change `--data_path` from `dataset` to `videos`
3. Remove `--base_model` parameter (not needed)
4. Add `--num_frames` parameter (default: 8)

**Video Organization:**

Ensure videos are in flat `videos/` directory:
```
videos/
├── 00000340.mp4
├── 00197706.mp4
└── ...
```

If your JSON has paths like `cat-cow_pose/00197706.mp4`, the script automatically extracts just the filename.

## Troubleshooting

**Empty predictions (all zeros in report):**
- Fixed by using correct tokenization with `apply_chat_template()`
- Ensure model is loaded with `FastVisionModel.for_inference(model)`

**Video not found errors:**
- Check videos are in `videos/` directory
- Script extracts filename from JSON path automatically

**Report generation fails:**
- Install missing package: `pip install evaluate`
- Check predictions JSON file exists and is valid

## Related Files

- Fine-tuning: `gemma3_finetune_unsloth.py`
- Single video inference: `utils/infer_qved.py`
- Evaluation script: `eval/eval_gemma3n.py`
- Training wrapper: `scripts/finetune_gemma3n_unsloth.sh`
