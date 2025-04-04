# Email Spam Classifier

This project is an **Email Spam Classifier** built using machine learning techniques. It allows users to classify email text as either "Spam" or "Not Spam" using different machine learning models.

## Features
- **Streamlit Web App**: A user-friendly interface to input email text and get predictions.
- **Machine Learning Models**:
  - Logistic Regression
  - Naïve Bayes
  - Decision Tree
- **TF-IDF Vectorization**: Converts email text into numerical features for model training and prediction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/leslyvj/email_spam_classifier.git
   cd email_spam_classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app in your browser.
2. Paste the email text into the input box.
3. Select a machine learning model from the sidebar.
4. Click the "Predict" button to classify the email as "Spam" or "Not Spam."

## Project Structure
```
spam classifier/
├── app.py                # Streamlit app for user interaction
├── train_models.py       # Script to train and save machine learning models
├── vectorizer.pkl        # Saved TF-IDF vectorizer
├── logistic_model.pkl    # Saved Logistic Regression model
├── naive_bayes_model.pkl # Saved Naïve Bayes model
├── decision_tree_model.pkl # Saved Decision Tree model
├── cleaned_spam_dataset.csv # Preprocessed dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Dataset
The project uses a cleaned dataset (`cleaned_spam_dataset.csv`) containing email messages and their corresponding labels (`Spam` or `Not Spam`).

## Dependencies
- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.

## Acknowledgments
- [NLTK](https://www.nltk.org/) for text preprocessing.
- [Scikit-learn](https://scikit-learn.org/) for machine learning models.
- [Streamlit](https://streamlit.io/) for building the web app.

---
Feel free to contribute to this project by submitting issues or pull requests!


