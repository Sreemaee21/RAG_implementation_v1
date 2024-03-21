import google.generativeai as genai
import lancedb
from database_creation import create_db, model, main

## setting the API key
genai.configure(api_key = "AIzaSyBJB02KG8EtDqnaIzElHrOyV6lUEVb5VdE")


## defining the model - here we use  gemini- pro becasue of it's text generation capabilities

model1 = genai.GenerativeModel('gemini-pro')
db_file_path = "/mnt/c/llm_SERC_documentation/sree_llm_chatbot/RAG_assignment/text_embeddings_db.lancedb"



### debug/test working:
# %%time
# response = model1.generate_content("How do I set up an AWS account?")
# print(response.prompt_feedback)
# print(response.text)




def create_prompt(query, top_results):
    limit = 3750

    prompt_start = (
        "Answer the question based only on the context below.   Do not use external knowledge from your databases.\n\n" +
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )

    # If no relevant contexts found, use the entire table
    if len(top_results) == 0:
        context_text = ""
    else:
        context_text = "\n\n---\n\n".join(top_results['text'].tolist())

    # Combine context and prompt
    prompt = prompt_start + context_text + prompt_end

    return prompt




def complete(prompt):
    response = model1.generate_content(prompt)
    return response.text



### Example usage:
# query = "who was the 12th person on the moon and when did they land?"
# response_text = complete(query)
# print(response_text)

def main(question):
    # query = "what are the pillars of aws"
    #  Connect to lancedb and get the existing created embeddings table
    db = lancedb.connect(db_file_path)
    tbl = db.open_table("text_embeddings_final")  # Assuming this is the correct table name

    # Search for relevant contexts based on the query by similarity matching the emebedding
    ## cosine similarity is beign used to match the query embedding to the closest matches in the lanceDB embedding
    test_emb = model.encode(question)
    top_results = tbl.search(test_emb).limit(3).to_pandas()

    # Create prompt based on top three results
    prompt = create_prompt(question, top_results)

    # Generate response based on the prompt
    response_text = complete(prompt)
    print(top_results)  ### Can comment if you don't want to see the top three embeddings matches
    print("Generated Response:")
    print(response_text)
    
    return response_text

    

if __name__ == "__main__":
    main()
