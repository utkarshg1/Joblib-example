# Iris Species Prediction App

This web application predicts the species of an iris flower based on its sepal and petal dimensions. It is built with Streamlit and uses a pre-trained machine learning model.

## Getting Started

### Prerequisites

- Python 3.12.4
- Streamlit
- Scikit-learn
- Joblib
- Pandas
- Docker (optional, for containerization)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/iris-prediction-app.git
   cd iris-prediction-app
   ```

2. **Create a virtual environment and activate it**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   ```bash
   streamlit run app.py
   ```

   Access the app at `http://localhost:8501`.

### Using Docker

To run the app with Docker:

1. **Build the Docker image**:

   ```bash
   docker build -t iris-prediction-app .
   ```

2. **Run the Docker container**:

   ```bash
   docker run -p 8501:8501 iris-prediction-app
   ```

   Access the app at `http://localhost:8501`.

## Usage

Once the app is running, input the sepal length, sepal width, petal length, and petal width. Click on "Predict" to get the predicted species and the corresponding probabilities.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- The Iris dataset is sourced from the UCI Machine Learning Repository.
- Streamlit is used for the UI, and Scikit-learn is used for the machine learning model.