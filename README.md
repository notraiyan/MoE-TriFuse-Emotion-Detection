# MoE-TriFuse: Emotion Recognition in Conversations (MELD)

Context-aware trimodal (text, audio, video) mixture-of-experts for emotion recognition on [MELD](https://github.com/declare-lab/MELD). Everything lives in one notebook, `meld-dataset.ipynb`.

## Pipeline

1. **Data**: labels and transcripts from the official MELD CSVs; video clips from `MELD.Raw.tar.gz` (~11 GB, downloaded once).
2. **Frozen features**
   - Text: RoBERTa-large with dialogue context, all layers kept.
   - Audio: emotion-tuned WavLM-large (MSP-Podcast), plus generic WavLM-large for ablations.
   - Video: speaking-face expressions (YuNet face detection, Light-ASD active speaker detection, HSEmotion AffectNet embeddings), plus CLIP ViT-B/16 for ablations.
3. **Fine-tuned text model**: RoBERTa-large or a LoRA-tuned decoder LLM (default `Qwen/Qwen3-8B`) using preceding and following turns, cross-fitted over 5 folds of the training dialogues. Out-of-fold predictions replace the frozen text features.
4. **MoE-TriFuse**: per-modality dialogue Transformers, a cross-modal fusion expert, and a gated mixture of text, audio, video and fusion experts. Missing modalities get a learned token and zero gate weight.
5. **Objective**: class-weighted focal loss, auxiliary expert losses, InfoNCE cross-modal alignment, modality dropout.
6. **Evaluation**: baselines (unimodal, early, late, cross-attention), ablations and missing-modality tests over 5 seeds, with tables, paired-bootstrap significance tests, gate analysis, confusion matrix and error examples.

## Repository layout

| Path | Description |
| --- | --- |
| `meld-dataset.ipynb` | Full pipeline: features, training, evaluation, figures |
| `roihu/setup_env.sh` | One-time environment setup on CSC Roihu (aarch64 GH200 nodes) |
| `roihu/meld.sbatch` | Slurm job that runs the notebook with papermill |

## Running on Kaggle / interactively

- Use a GPU accelerator (T4 x2; one GPU is used) with internet enabled.
- Run once with `DEBUG = True` (25 dialogues per split, 1 seed, few epochs) to check everything.
- Set `DEBUG = False` and run the full notebook. Estimated 6-8 h; finished runs and folds are skipped on re-run.

## Running on CSC Roihu

Run from the folder containing `meld-dataset.ipynb` and `roihu/`. Replace `project_2001234` with your project.

```bash
# 1. One-time: build the virtual environment in /scratch/<project>/<user>/meld-venv
sbatch --account=project_2001234 roihu/setup_env.sh project_2001234

# 2. Quick check (MELD_DEBUG=1)
sbatch --account=project_2001234 --time=01:00:00 roihu/meld.sbatch debug

# 3. Full run (Qwen/Qwen3-8B)
sbatch --account=project_2001234 roihu/meld.sbatch

# Another fine-tuned text model
sbatch --account=project_2001234 roihu/meld.sbatch Qwen/Qwen3-14B
```

- Job log: `meld-<jobid>.out`; executed notebook: `/scratch/<project>/<user>/meld/meld-dataset_<jobid>.ipynb`.
- MELD clips, model weights, features and finished runs are cached under `/scratch/<project>/<user>/meld`, so resubmitting continues where a job stopped.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `MELD_FT_MODEL` | Hugging Face model for the fine-tuned text classifier |
| `MELD_WORK_DIR` | Features, runs and results |
| `MELD_DOWNLOAD_DIR` | Raw MELD video clips |
| `HF_HOME` | Hugging Face model cache |
| `MELD_DEBUG` | Set to `1` for a quick debug run |
