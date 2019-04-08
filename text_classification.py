import tensorflow as tf
import my_model
import load_data

data = load_data.load_imb_sentiment_analysis_dataset("data/")

model = my_model.train_ngram_model(data)
