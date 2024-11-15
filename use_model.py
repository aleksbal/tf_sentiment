import tensorflow as tf
from transformers import AutoTokenizer

# Load the saved model
model = tf.keras.models.load_model("saved_model/my_text_classifier")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def predict(texts):
    # Tokenize the input text
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")
    
    # Make predictions
    predictions = model(inputs)
    
    # Get the predicted labels
    predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()
    
    return predicted_labels

# Example text inputs
texts = ["The movie was amazing!", "I really disliked the plot and characters."]

# Predict sentiment
predicted_labels = predict(texts)

# Output predictions
for text, label in zip(texts, predicted_labels):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"Text: {text} | Sentiment: {sentiment}")