## Project Overview

This project builds a **few-shot multimodal classifier** that integrates both product images and metadata text to categorize fashion products. It leverages the OpenAI CLIP model as a frozen backbone for feature extraction and trains a lightweight classifier head on top, making it suitable for low-data regimes.

The classifier seamlessly blends visual and textual information to improve accuracy, demonstrating modern AI techniques for handling real-world messy, multimodal data with limited samples per class.

***

## What This Project Does

- Uses the **CLIP** model to encode images and text into shared embedding space.
- Freezes the CLIP backbone and trains a custom classifier head to combine both modalities.
- Handles **few-shot** scenarios by training on very small labeled samples per category (5–10 samples per class).
- Trains and evaluates performance on a subset of the Fashion Product Images + Metadata dataset.
- Provides an **interactive Gradio demo** to upload product images and metadata text for instant category predictions.

***

## Project Completeness

- Dataset downloading, exploration, and preprocessing included.
- Model building, training, and evaluation loops implemented.
- Model saving/loading and checkpoint management present.
- Ablation study framework setup via modular classifier classes.
- Gradio demo app built for end-user interaction.
- Ready for reproducible training and testing in Google Colab.

***

## How to Test The Project

1. Clone or download this repository.
2. Open the notebook `few_shot_multimodal_classifier_image_text.ipynb` in Google Colab.
3. Run all cells sequentially to:
   - Download and preprocess the dataset.
   - Load and freeze CLIP model.
   - Define and train the classifier head.
   - Evaluate on validation data.
4. After training, launch the **Gradio demo** to upload new product images and enter text metadata.
5. Observe interactive category predictions with confidence scores.
6. Optionally, retrain with different data splits or longer epochs for better accuracy.

***

## How to Showcase Screenshot of Results

- During notebook execution, display sample input images alongside predicted category labels and confidence scores using Matplotlib or IPython display.
- Alternatively, take screenshots of the Gradio web demo running:
  - Use OS native screenshot tools or Chrome's Developer Tools device toolbar.
  - Save screenshots showing sample inputs and model outputs.
- Post these screenshots in this README under a "Results" section for visual proof of model capability.

Example code snippet in notebook for displaying results inline:

```python
import matplotlib.pyplot as plt

def show_results(image, predicted_label, confidence):
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()

# Example usage after inference:
# Assuming 'img' is PIL image, 'label' is string, 'conf' is float
show_results(img, label, conf)
```

***

## Technologies and Libraries Used

- PyTorch & Torchvision for model and training
- OpenAI CLIP for multimodal feature extraction
- PEFT (optional) for parameter-efficient fine-tuning (not used now)
- Gradio for web GUI demo
- Pandas & NumPy for data handling
- Matplotlib for result visualization

***

## Future Work and Improvements

- Incorporate LoRA fine-tuning when compatible CLIP backbones are available.
- Add support for more complex downstream networks on embeddings.
- Extend to few-shot learning meta-learning methods.
- Implement batch inference and multi-modal retrieval.
