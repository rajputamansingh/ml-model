# Brain Disorder ML Analysis App

A Streamlit web application for comparative analysis of machine learning models on brain MRI images for disorder detection.

## Features

- Upload brain MRI images for analysis
- Compare multiple ML model architectures (ResNet-50, VGG-16, Inception V3, AlexNet)
- Analyze across different brain disorders (Alzheimer's, Brain Tumors, Parkinson's)
- Visualize performance metrics including accuracy, precision, recall, F1-score
- View confusion matrices and class probabilities
- Compare models across different disorders

## Deployment Instructions for Render

This application is configured for easy deployment on Render.com.

### How to deploy:

1. Fork or clone this repository to your GitHub account
2. Sign up for a Render account if you don't have one
3. In Render dashboard, click on "New Web Service"
4. Connect your GitHub account and select this repository
5. Configure the service with the following settings:
   - **Name**: brain-disorder-ml-analysis (or any name you prefer)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
6. Click "Create Web Service"

The deployment process will start automatically. Once completed, Render will provide you with a URL to access your application.

## Local Development

### Prerequisites:
- Python 3.10 or higher
- pip package manager

### Setup:

1. Clone this repository
   ```
   git clone https://github.com/yourusername/brain-disorder-ml-analysis.git
   cd brain-disorder-ml-analysis
   ```

2. Create and activate a virtual environment (optional but recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Run the application
   ```
   streamlit run app.py
   ```

5. Open your browser and navigate to http://localhost:8501

## Project Structure

- `app.py` - Main Streamlit application file
- `requirements.txt` - Python dependencies
- `Procfile` - Instructions for deployment
- `runtime.txt` - Python runtime specification
- `samples/` - Directory for sample MRI images (created on first run)

## Future Improvements

- Connect with real trained models for each disorder
- Add explainability features to show which regions influence predictions
- Implement database storage for historical analyses
- Add batch processing for multiple images
- Expand to more brain disorders and model architectures

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Developed by Aman Singh  
Under the Supervision of Mr. Ashis Datta and Dr. Palash Ghosal
