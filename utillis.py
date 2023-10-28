from sentence_transformers import SentenceTransformer
import pinecone
import streamlit as st
model = SentenceTransformer('all-MiniLM-L6-v2')


pinecone.init(api_key='fd9453a9-749a-4400-9673-053edbbe70a7', environment='gcp-starter')
index = pinecone.Index('prototype')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string