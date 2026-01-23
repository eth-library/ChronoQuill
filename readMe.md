# ChronoQuill

ChronoQuill's transformation pipeline leverages AI-powered HTR, layout classification, and few-shot learning to convert handwritten documents into structured Markdown.

<div style="text-align: center;"> <img src="supplements/ChronoQuill.png" alt="chronoQuill"> </div>

## Setup Instructions

```bash
git clone git@github.com:eth-library/ChronoQuill.git
cd ChronoQuill
```

### Environment and Libraries
```bash
uv venv chrono
source chrono/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install google-genai timm dotenv
```

### Environment Variables
Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

### Classifier & Few-Shot Samples
On the top level, populate the following folders.
```bash
# The classifier model:
mkdir models && cd models/
wget https://polybox.ethz.ch/index.php/s/Je9JEwST2drDp4K/download
unzip download

# Ground Truth samples for Few-Shot Learning:
mkdir few_shot & cd few_shot/
wget https://polybox.ethz.ch/index.php/s/5kSGRHYmz2m4tCE/download
unzip download

# Input/Output
mkdir data
```

## Project Structure
- `chrono_quill.py` — Main pipeline script
- `utils.py` — Utility functions and helper classes
- `prompts.py` — System prompts for Gemini API
- `few_shot/` — Few-shot ground truth samples
- `models/` — Pretrained model files
- `data/` — Input and output data

## Transform TIFF & JPG into Markdown
```bash
python chrono_quill.py
```

## License
Apache 2.0

## References
- [Google GenAI](https://ai.google.dev/)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://huggingface.co/timm/swin_large_patch4_window7_224.ms_in22k)

## Remarks
The pipeline is specialized to process ETH's school council protocols. For different use cases, consider pretraining your own classifier and provide suitable grount truth for few-shot learning.

## BibTeX
```bash
@article{marbach2026closed,
  title={Closed-Vocabulary Multi-Label Indexing Pipeline for Historical Documents},
  author={Marbach, Jeremy},
  year={2026},
  publisher={ETH Zurich}
}
```