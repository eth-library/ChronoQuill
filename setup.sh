#!/bin/bash
# setup.sh - Download models and data

mkdir -p models data
cd models
wget -O layout_classifier_model.pt "https://polybox.ethz.ch/index.php/s/Je9JEwST2drDp4K/download"
cd ..
wget https://polybox.ethz.ch/index.php/s/5kSGRHYmz2m4tCE/download
unzip download && rm download