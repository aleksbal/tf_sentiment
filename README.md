
# Text Classification Project with TensorFlow and Hugging Face Transformers

This project is a text sentiment classification tool built with TensorFlow and Hugging Face’s Transformers library. It demonstrates how to train a binary text classifier on the IMDb movie review dataset, save the trained model, and use it manually for predictions. The example provided is based on Transformers, specifically using the DistilBERT model, which belongs to the Transformer architecture family.

## Project Structure

```
text_classification_project/
├── env/                     # Virtual environment directory
├── main.py                  # Main script for training and saving the model
├── use_model.py             # Script for loading the model and making predictions
├── requirements.txt         # File to track dependencies
└── README.md                # Project documentation
```

## Requirements

- Python 3.8 - 3.10
- TensorFlow
- Hugging Face Transformers
- Datasets (from Hugging Face)

## Setup Instructions

### 1. Clone the Repository

Clone the project to your local machine:

```bash
git clone https://github.com/your_username/tf_txt_classif.git
cd text_classification_project
```

### 2. Set Up a Virtual Environment

Create and activate a virtual environment to keep dependencies isolated:

```bash
# Create a virtual environment with Python 3.10
python3.10 -m venv env

# Activate the virtual environment
# On macOS/Linux
source env/bin/activate
# On Windows
.\env\Scriptsctivate
```

### 3. Install Dependencies

With the virtual environment active, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Install Project Dependencies

If not yet listed in `requirements.txt`, you may need to install specific packages:

```bash
pip install tensorflow transformers datasets
```

## Training the Model

1. Run the `main.py` script to train the model on the IMDb movie review dataset and save the model.

```bash
python main.py
```

### Script Breakdown - `main.py`

- **Data Loading**: Loads the IMDb dataset from Hugging Face.
- **Tokenizer**: Tokenizes the input text using a pre-trained DistilBERT tokenizer.
- **Model**: Initializes and fine-tunes a DistilBERT model for binary text classification.
- **Training**: Compiles and trains the model, then saves it to `saved_model/my_text_classifier`.

Here’s the main structure of `main.py`:

```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import tensorflow as tf

# Load the IMDb dataset
dataset = load_dataset("imdb")
train_data, test_data = dataset["train"], dataset["test"]

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize the data
def tokenize_data(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

train_data = train_data.map(tokenize_data, batched=True)
test_data = test_data.map(tokenize_data, batched=True)

# Convert to TensorFlow format
train_data = train_data.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=32
)

test_data = test_data.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=32
)

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(train_data, validation_data=test_data, epochs=3)

# Save the model
model.save("saved_model/my_text_classifier")
```

## Using the Model for Predictions

After training, you can use `use_model.py` to load the saved model and predict sentiment on new text inputs.

### Script Breakdown - `use_model.py`

- **Load the Model**: Loads the saved model from `saved_model/my_text_classifier`.
- **Tokenize New Inputs**: Uses the same tokenizer to preprocess new text.
- **Predict Sentiment**: Classifies text as positive or negative sentiment.

### Example - `use_model.py`

```python
import tensorflow as tf
from transformers import AutoTokenizer

# Load the saved model and tokenizer
model = tf.keras.models.load_model("saved_model/my_text_classifier")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def predict(texts):
    # Tokenize input text
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    # Predict sentiment
    predictions = model(inputs)
    predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()
    return predicted_labels

# Test the model with new inputs
texts = ["The movie was amazing!", "I really disliked the plot."]
predicted_labels = predict(texts)

for text, label in zip(texts, predicted_labels):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"Text: {text} | Sentiment: {sentiment}")
```

### Running the Prediction Script

Run `use_model.py` to see predictions for sample texts:

```bash
python use_model.py
```

Expected output:

```
Text: The movie was amazing! | Sentiment: Positive
Text: I really disliked the plot. | Sentiment: Negative
```

## Deactivating the Environment

When you’re finished, deactivate the virtual environment with:

```bash
deactivate
```

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [Transformers Documentation](https://huggingface.co/transformers/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

## License

This project is licensed under the MIT License. 
