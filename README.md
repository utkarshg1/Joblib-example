# Iris Species Prediction App

This web application predicts the species of an iris flower based on its sepal and petal dimensions. It is built with Streamlit and uses a pre-trained machine learning model.

## Demo App Link

URL - [https://joblib-example.onrender.com/](https://joblib-example.onrender.com/)

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
   git clone https://github.com/utkarshg1/Joblib-example.git
   cd Joblib-example
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
   python3 -m streamlit run app.py
   ```

   Access the app at `http://localhost:8501`.

### Using Docker

To run the app with Docker:

1. **Run the Docker container**:

   ```bash
   docker compose up --build
   ```

   Access the app at `http://localhost:8501`.

## Usage

Once the app is running, input the sepal length, sepal width, petal length, and petal width. Click on "Predict" to get the predicted species and the corresponding probabilities.

## Dockerhub link

Dockerhub Link - [https://hub.docker.com/r/utkarshg1/streamlit-iris](https://hub.docker.com/r/utkarshg1/streamlit-iris)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- The Iris dataset is sourced from the UCI Machine Learning Repository.
- Streamlit is used for the UI, and Scikit-learn is used for the machine learning model.
