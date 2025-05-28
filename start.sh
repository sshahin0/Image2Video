#!/bin/bash

echo "Starting FlashAttention setup..."
cd /app/flash-attention && python setup.py install || echo "FlashAttention already installed."

echo "Starting Gradio interface..."
cd /app
python gradio_app.py
