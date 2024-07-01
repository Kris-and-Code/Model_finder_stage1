import os
import tensorflow as tf
from transformers import pipeline, TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate

# Set TensorFlow logging level to ERROR to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Define models to evaluate
model_names = [
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased",
    "joeddav/xlm-roberta-large-xnli",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "bhadresh-savani/bert-base-multilingual-cased-sentiment"
]

# Load dataset and create a subset
dataset = load_dataset("amazon_polarity", split="test")
subset_size = 1000  # Define the size of the subset
dataset = dataset.shuffle(seed=42).select(range(subset_size))

# Metric for evaluation
metric = evaluate.load("accuracy", trust_remote_code=True)

# Function to process the dataset in batches
def evaluate_model_on_dataset(nlp, dataset, batch_size=32):
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        texts = batch['content']
        labels = batch['label']
        
        predictions = nlp(texts)
        for prediction, label in zip(predictions, labels):
            pred_label = 1 if 'POSITIVE' in prediction['label'] else 0
            if pred_label == label:
                correct_predictions += 1
            total_predictions += 1
        
        print(f"Processed {min(i+batch_size, len(dataset))}/{len(dataset)} examples")
    
    return correct_predictions, total_predictions

# Evaluate each model
for model_name in model_names:
    try:
        print(f"Evaluating model: {model_name}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline for sentiment analysis
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, framework='tf')
        
        # Evaluate model on dataset in batches
        correct_predictions, total_predictions = evaluate_model_on_dataset(nlp, dataset, batch_size=32)
        
        # Calculate and print accuracy
        accuracy = correct_predictions / total_predictions
        print(f"Model: {model_name}, Accuracy: {accuracy}")
    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
