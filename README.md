# MNIST Handwritten Digit Generation Web App

This project provides a web application that generates images of handwritten digits (0–9) using a model trained from scratch on the MNIST dataset. The app allows users to select a digit and generates 5 diverse images of that digit, similar to the MNIST style.

## Features
- Select a digit (0–9) to generate
- Generates 5 diverse images per digit
- Images are displayed in a grid, MNIST-style
- Model trained from scratch (no pre-trained weights)

## File Structure
```
MNISTwebpage/
│
├── app.py                # Streamlit web app
├── train_cvae.py         # Model training script (PyTorch, CVAE)
├── cvae_model.pth        # Trained model weights (generated after training)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model:**
   ```bash
   python train_cvae.py
   ```
   This will save `cvae_model.pth` in the project directory.
4. **Run the web app:**
   ```bash
   streamlit run app.py
   ```

## Notes
- Model training is done from scratch using PyTorch and the MNIST dataset.
- The web app uses the trained model to generate digit images on demand.
- For deployment, you can use Streamlit Community Cloud or similar services.

## License
MIT 