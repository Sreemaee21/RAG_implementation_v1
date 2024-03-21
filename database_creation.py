
import os
import lancedb
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


# Initialize SentenceTransformer model
model_name = 'paraphrase-MiniLM-L6-v2'  # You can choose other models from https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer(model_name)


### trying paragraph splitting instead of chunks of limited numeber of charecters
    
def split_text(text, max_chunk_size=500, max_para_length=600):
    # Split text into paragraphs
    paragraphs = text.split('\n\n')  # Assuming paragraphs are separated by two newline characters
    # Initialize variables
    chunks = []
    current_chunk = ""
    # Split paragraphs into chunks of approximately equal size
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 < max_chunk_size:  # Adding 2 for the newline characters
            current_chunk += para + '\n\n'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + '\n\n'
    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks



def creating_chunks(file_path):

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text_content = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        text_content = ""
    text_chunks = split_text(text_content)
    return text_chunks



def create_embeddings(text_chunks):
# Create embeddings for each text chunk
    embeddings = []
    for chunk in text_chunks:
        chunk_embeddings = model.encode(chunk)
        embeddings.append(chunk_embeddings)
    return embeddings


def create_db(text_chunks, embeddings, db_file_path):
    df = pd.DataFrame(
        {
            "idx": range(len(text_chunks)),
            "text": text_chunks,
            "vector": embeddings
        }
    )
    db = lancedb.connect(db_file_path)
    tbl = db.create_table("text_embeddings_final", data=df)
    return tbl




def main():
    file_path = "/mnt/c/llm_SERC_documentation/sree_llm_chatbot/RAG_assignment/aws_text.txt"  # Replace with your file path
    db_file_path = "/mnt/c/llm_SERC_documentation/sree_llm_chatbot/RAG_assignment/text_embeddings_db.lancedb"
    text_chunks = creating_chunks(file_path)
    embeddings = create_embeddings(text_chunks)
    tbl = create_db(text_chunks, embeddings, db_file_path)
    print(f"Number of embeddings created: {len(tbl)}")

    # Uncomment and provide necessary arguments for testing retrieval
    # test_query = 'How do I set up an AWS account?'
    # test_emb = model.encode(test_query)
    # testing_retrieval(test_query, test_emb, tbl)

if __name__ == "__main__":
    main()


## Optional: Print number of embeddings created
# print(f"Number of embeddings created: {len(tbl)}")

### only for debugging
### Print the embeddings and their shapes for debugging
# for idx, embedding in enumerate(embeddings):
#     print(f"Embedding {idx}:")
#     print(embedding)
#     print(f"Shape: {embedding.shape}")

### test the retrieval of info from lanceDB and check for closest embedding match 

# def testing_retrieval(test_query, test_emb, tbl):
#     test_query = 'How do I set up an AWS account?'
#     test_emb = model.encode(test_query)
#     tbl.search(test_emb).limit(3).to_pandas()




# http://127.0.0.1:5000/api/question