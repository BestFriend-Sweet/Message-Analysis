from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import gradio as gr
import json
import tiktoken
import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
from metadata import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME

tokenizer = tiktoken.get_encoding('p50k_base')

batch_limit = 5


def get_json_text(files):
    data = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as input_file:
            data += json.load(input_file)
    return data


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def get_text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=0,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    return text_splitter


def get_embeddings():
    model_name = 'text-embedding-ada-002'

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                                  document_model_name=model_name, query_model_name=model_name)

    return embeddings


def create_pinecone_index():
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    pinecone.create_index(
        name=INDEX_NAME,
        metric='dotproduct',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )


def upsert_pinecone(data, text_splitter, embeddings, index):
    texts = []
    metadatas = []

    for i, record in enumerate(tqdm(data)):
        metadata = {
            'metadata': record['metadata']
        }

        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['content'])

        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]

        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)

        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embeddings.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []


def similarity_search(vectorstore, query):
    return vectorstore.similarity_search(query, k=3)


def generative_answering(query):
    global chain
    
    print(query)

    return chain.run(query.replace('Kim', 'Tomm')).replace('Tomm', 'Kim')


def main():
    global chain

    data = get_json_text(['processed_data/messages.json'])

    text_splitter = get_text_splitter()
    embeddings = get_embeddings()

    # create_pinecone_index() # Create Pinecone Index
    
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index = pinecone.Index(INDEX_NAME)
    
    # upsert_pinecone(data, text_splitter, embeddings, index)  # Insert data to Pinecone Database

    # Create Vectorstore
    text_field = "text"
    vectorstore = Pinecone(
        index, embeddings.embed_query, text_field
    )

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name='gpt-4-0613')
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    iface = gr.Interface(fn=generative_answering, inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                         outputs="text", title="Custom AI Chatbot")
    iface.launch(share=True)

    # query = input("Input Query: ")
    #
    # search_results = similarity_search(vectorstore, query)
    #
    # generative_answer = generative_answering(query)
    #
    # print("Similarity Search Result: ", search_results)
    # print("________________")
    # print("Generative Answer: ", generative_answer)


if __name__ == '__main__':
    main()
