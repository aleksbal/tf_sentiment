from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
import tensorflow as tf
import platform

tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load data
dataset = load_dataset("imdb")
train_data, test_data = dataset["train"], dataset["test"]

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize and prepare data
def tokenize_data(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

train_data = train_data.map(tokenize_data, batched=True)
test_data = test_data.map(tokenize_data, batched=True)

# Convert data to TensorFlow format
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

if platform.system() == "Darwin" and platform.processor() == "arm":
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)
else:
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

# Compile and train model
model.compile(opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(train_data, validation_data=test_data, epochs=3)

# Save the entire model
model.save("saved_model/my_text_classifier")