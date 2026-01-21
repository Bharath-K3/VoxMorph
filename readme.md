# VoxMorph: Scalable Zero-Shot Voice Identity Morphing via Disentangled Embeddings

[![Conference](https://img.shields.io/badge/ICASSP-2026-blue)](https://2026.ieeeicassp.org/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation for our paper, **"VoxMorph: Scalable Zero-Shot Voice Identity Morphing via Disentangled Embeddings"**, accepted at the **IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2026**.

**Authors:** Bharath Krishnamurthy and Ajita Rattani  
*University of North Texas*

---

<p align="center">
  <img src="Assets/VoxMorph_Teaser.jpg" width="800" alt="VoxMorph Architecture">
  <br>
  <em>Figure 1: Architectural overview of the VoxMorph framework. The process consists of three core stages: (1) Extraction of disentangled prosody and timbre embeddings; (2) Independent interpolation via Spherical Linear Interpolation (Slerp); and (3) Synthesis via an autoregressive language model and Conditional Flow Matching (CFM) network.</em>
</p>

## Abstract

> Morphing attacks threaten biometric security by creating synthetic samples that can impersonate multiple individuals. While extensively studied for face recognition, this vulnerability remains largely unexplored for voice biometrics. The only prior work on voice morphing is computationally expensive, non-scalable, and restricted to acoustically similar identity pairs, limiting its practicality. We propose **VoxMorph**, a novel zero-shot framework that generates high-fidelity voice morphs from as little as five seconds of audio per subject, without model retraining. Our approach disentangles vocal characteristics into prosody and timbre embeddings, enabling fine-grained interpolation of speaking style and identity. These embeddings are blended via Spherical Linear Interpolation (Slerp) and synthesized through an autoregressive language model (LM) together with a Conditional Flow Matching (CFM) network. VoxMorph achieves SOTA results, outperforming existing methods with a **2.6× improvement in audio quality** and a **67.8% morphing attack success rate** (FMMPMR) at strict security thresholds.

---

## Key Contributions

-   **Zero-Shot Framework**: VoxMorph eliminates the need for identity-specific fine-tuning, generating morphs from just 5 seconds of source audio.
-   **Disentangled Representation**: We decouple vocal characteristics into **Prosody** (style) and **Timbre** (identity) embeddings, allowing for granular, artifact-free interpolation.
-   **Spherical Linear Interpolation (Slerp)**: We demonstrate that Slerp is mathematically superior to linear averaging for maintaining the geometric structure of speaker embeddings during fusion.
-   **State-of-the-Art Performance**: We establish new benchmarks in voice morphing, achieving superior audio quality (FAD: 4.90) and unprecedented attack success rates (FMMPMR: 67.80% @ 0.01% FAR).

## Performance Comparison

VoxMorph consistently outperforms state-of-the-art voice conversion and morphing baselines across audio quality, intelligibility, and attack success metrics.

| Method | FAD ↓ (Quality) | WER ↓ (Intelligibility) | KLD ↓ (Spectral) | MMPMR (%) @ 0.01% | FMMPMR (%) @ 0.01% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ViM (Baseline)** | 7.52 | 1.06 | 0.3501 | 2.61 | 0.00 |
| **Vevo (Zero-Shot)** | 9.14 | 0.54 | 0.1899 | - | - |
| **MorphFader** | 8.96 | 1.84 | 0.4332 | - | - |
| **VoxMorph (Ours)** | **4.90** | **0.19** | **0.1385** | **99.80** | **67.80** |

*Table 1: Quantitative comparison against SOTA methods. VoxMorph demonstrates superior fidelity (lower FAD/KLD), near-perfect intelligibility (lower WER), and a high threat potential (MMPMR/FMMPMR).*

---

## Installation

Follow these instructions to set up a dedicated environment for running the VoxMorph inference engine.

### Prerequisites

*   **Anaconda or Miniconda**: Required for environment management.
*   **Git**: To clone the repository.
*   **NVIDIA GPU**: Recommended for faster inference (CUDA support).

### Step-by-Step Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Bharath-K3/VoxMorph.git
    cd VoxMorph
    ```

2.  **Create and Activate Conda Environment**
    We recommend using Python 3.11 for compatibility with the backend TTS modules.

    ```bash
    # Create environment 'voxmorph'
    conda create -n voxmorph python=3.11 -y

    # Activate the environment
    conda activate voxmorph
    ```

3.  **Install Dependencies**
    Install the required libraries, including Torch, Gradio, and audio processing tools.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Directory Structure**
    Ensure your directory looks like this before running the code. We have provided sample assets in the repository.
    ```
    VoxMorph/
    ├── Assets/
    │   ├── Audio1.flac
    │   ├── Audio2.flac
    │   └── VoxMorph_Teaser.jpg
    ├── chatterbox/
    ├── data/
    │   ├── speaker_1_dir_30/
    │   └── speaker_2_dir_6000/
    ├── encoders/
    │   ├── __init__.py
    │   ├── ecapa.py
    │   ├── hubert.py
    │   └── wav2vec2.py
    ├── experiment_1/
    ├── outputs/
    ├── app.py
    ├── config.yaml
    ├── inference.py
    ├── readme.md
    ├── requirements.txt
    ├── utils.py
    └── VoxMorph.py
    ```

---

## Usage

We provide two modes of inference: a graphical web interface (Gradio) and a command-line interface (CLI) for batch processing.

### 1. Graphical Web Interface (Gradio)
This is the recommended method to visualize the morphing process interactively.

```bash
python app.py
```

Once running, the terminal will provide a URL (e.g., http://127.0.0.1:7860).

1. Upload two audio files (or use microphones).
2. Adjust the Alpha Slider (0.0 to 1.0) to control the morphing ratio.
3. Click *Perform Zero-Shot Morphing*.

### 2. Command Line Interface (CLI)
For headless environments or automation, use the CLI script.

**Basic Usage (Uses default files in `Assets/`):**

```bash
python inference.py
```

**Custom Arguments**
```bash
python inference.py --source_a "path/to/speaker_A.wav" --source_b "path/to/speaker_B.wav" --alpha 0.5 --text "This is a synthetic voice morph."
```

| Argument     | Description                           | Default              |
| :----------- | :------------------------------------ | :------------------- |
| `--source_a` | Path to the first speaker audio       | `Assets/Audio1.flac` |
| `--source_b` | Path to the second speaker audio      | `Assets/Audio2.flac` |
| `--alpha`    | Interpolation factor (0.0 - 1.0)      | `0.5`                |
| `--text`     | The text content to be synthesized    | Default string       |
| `--output_dir`| Directory to save results            | `outputs/`           |

### 3. Advanced Inference (VoxMorph.py)

For research experiments requiring robust data handling, use `VoxMorph.py`.

**Features:**
1.  **Multi-Shot Profiling:** You can pass a **directory** of audio clips instead of a single file. The script automatically consolidates them to create a stable speaker profile.
*   **Triplet Output:** Automatically generates the clone of Source A, Clone of Source B, and the Morphed result for direct comparison.
*   **Dynamic Encoder Switching:** Toggle between internal encoders (Default, ECAPA, Wav2Vec2, HuBERT) directly via CLI arguments or via `config.yaml`.

| Argument       | Description                                              | Options                                    |
| :------------- | :------------------------------------------------------- | :----------------------------------------- |
| `--source_a`   | Path to a file OR a directory containing clips for Spk A | File path or Folder path                   |
| `--source_b`   | Path to a file OR a directory containing clips for Spk B | File path or Folder path                   |
| `--alpha`      | Interpolation factor (0.0 = A, 1.0 = B)                  | `0.0` - `1.0`                              |
| `--encoder`    | Override the internal speaker encoder                    | `default`, `ecapa`, `wav2vec2`, `hubert`   |
| `--output_dir` | Directory to save the triplet results                    | Defaults to `results/`                     |

**Usage:**

```bash
# Using directories as input (The script handles the consolidation)
python VoxMorph.py --source_a "data/speaker_1_dir_30" --source_b "data/speaker_2_dir_6000" --alpha 0.5 --output_dir "experiment_1"
```

```bash
# Using ECAPA-TDNN Encoder
python VoxMorph.py --source_a "data/speaker_1_dir_30" --source_b "data/speaker_2_dir_6000" --alpha 0.5 --encoder ecapa --output_dir "experiment_ECAPA"
```

## Citation

If you find this work useful in your research, please consider citing our ICASSP 2026 paper:

```bibtex
@inproceedings{krishnamurthy2026voxmorph,
  title={VoxMorph: Scalable Zero-Shot Voice Identity Morphing via Disentangled Embeddings},
  author={Krishnamurthy, Bharath and Rattani, Ajita},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## Acknowledgements

We heavily drew inspiration from and built upon several pioneering open-source projects in the speech synthesis community. We specifically thank Resemble AI, CosyVoice, and Llama for their open contributions to the field. We also acknowledge the broader open-source community for providing the foundational tools and models that made this research possible.