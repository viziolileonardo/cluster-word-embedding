import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# Load the Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load your CSV file into a DataFrame
csv_path = "/Users/vizioli/git/teya/cluster-word-embedding/Caselist_1.csv"
data = pd.read_csv(csv_path)

# Get the text from the "Internal CR Agents Notes" column
notes = data["Internal Notes (Case)"].tolist()

# Generate embeddings for the notes
note_embeddings = embed(notes)

# Create a new DataFrame to hold the embeddings
embedding_df = pd.DataFrame(note_embeddings.numpy(), columns=[f"embedding_{i}" for i in range(note_embeddings.shape[1])])

# Concatenate the embedding DataFrame with the original DataFrame
result_df = pd.concat([data, embedding_df], axis=1)

# Save the updated DataFrame to a new CSV file
output_csv_path = "generalquestionswithembeddings.csv"
result_df.to_csv(output_csv_path, index=False)
