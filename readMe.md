# ChronoQuill

ChronoQuill is a pipeline for processing handwritten document images, performing HTR, layout classification, semantic segmentation, and generating refined Markdown files using zero-shot and few-shot learning.

<div style="text-align: center;"> <img src="supplements/ChronoQuill.png" alt="Rescue Mission"> </div>

## Features
- German-optimized HTR and semantic segmentation
- Layout classification using deep learning
- Zero-shot and few-shot Markdown generation
- Automated post-processing for margins
- Batch processing with parallelization

## Setup Instructions

### 1. Create and Activate Conda Environment
```bash
conda create --name chronoquill
conda activate chronoquill
```

### 2. Install Python Packages
```bash
conda install pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install google-genai
pip install timm
pip install dotenv
```

### 3. Environment Variables
Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage
Provide .tif or .jpg files to the `data/` folder.

Run ChronoQuill:
```bash
python chrono_quill.py
```

## Project Structure
- `chrono_quill.py` — Main pipeline script
- `utils.py` — Utility functions and helper classes
- `prompts.py` — System prompts for Gemini API
- `few_shot/` — Few-shot ground truth samples
- `models/` — Pretrained model files
- `data/` — Input and output data

Our pretrained layout classifier and few-shot samples can be downloaded [here](tobeadded).

## Requirements
- Python 3.13
- CUDA-enabled GPU (for Torch with cu128)

## License
MIT

## Acknowledgements
- [Google GenAI](https://ai.google.dev/)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://huggingface.co/timm/swin_large_patch4_window7_224.ms_in22k)

## Remarks
The pipeline is specialized to process ETH's school council protocols. For different use cases, consider pretraining your own classifier and provide suitable grount truth for few-shot learning.