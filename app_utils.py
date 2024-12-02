#!/usr/bin/env python3
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from transformers import set_seed

import time
import re
import os
from typing import List
from pathlib import Path

prompt_template = """You are an assistant to retrieve information from set of documents that belongs to individuals.
                    In the documents there are questions and answers separated with "|". Create a final answer to a user question by scanning the documents.
                    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
                    ALWAYS return a "SOURCES" part in your answer.
                                    
USER QUESTION: {question}
=========
{source_documents}
=========
FINAL ANSWER:"""

prompt_template_flascard = """You are an assistant to retrieve information from a document that belongs to an individual.
                                In the document there are questions and answers separated with "|". You will be given one UUID and corresponding document create a flash card for that individual.
                                In the flash card include every detail about the person in bullet points. This details will be used the guess the name of the person
                                Only include details from the document.
                                    
UUID: {question}
=========
{source_documents}
=========
FINAL ANSWER:"""

#set_seed(42)
    
def get_uuid_list_from_files(directory):
    # List all files in the given directory
    files = os.listdir(directory)
    # Extract the UUID part of the filename (remove the .txt extension)
    uuids = [file.split('.')[0] for file in files if file.endswith('.txt')]
    return uuids

def generate_answer(query, llm, db, embeddings, chatbot = False): 
    start = time.time()
    
    source_docs_path = 'source_data/individual_texts'
    
    uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'

    if not chatbot:
        filter = {}
        for match in re.finditer(uuid_pattern, query):
            uuid_id = match.group(0)
            filter = {'source': f'{source_docs_path}/{uuid_id}.txt'}
        docs_with_score = db.similarity_search_with_score(query,filter=filter)

        docs = [doc for doc in docs_with_score ]
        doc_page_contens = [doc[0] for doc in docs_with_score]

        ids = [str(i) for i in range(1, len(docs) + 1)]
        
         # We do this because we dont want to accumulate same filtered documents
        db_filtered = Chroma.from_documents(doc_page_contens, embeddings, ids=ids) 
        retriever_filtered = db_filtered.as_retriever(search_kwargs={"k": 100})
        
        retriever_final = retriever_filtered

    retriever = db.as_retriever(search_kwargs={"k": 8})
    
    if chatbot:
        
        PROMPT = ChatPromptTemplate.from_template(prompt_template) 
        
        print('chatbot is used!!!')
        retriever_final = retriever
        
    else:
        PROMPT = ChatPromptTemplate.from_template(prompt_template_flascard)
        

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"Content: {doc.page_content}\nSource: {doc.metadata['source']}" for doc in docs
        )
        
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            source_documents=(lambda x: format_docs(x["source_documents"]))
        )
        | PROMPT
        | llm
        | StrOutputParser()
    )
    rag_chain = RunnableParallel(
        {
            "source_documents": retriever_final,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    answer = rag_chain.invoke(query)

    end = time.time()

    # Print the result
    print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer['answer'])

    # Print the relevant sources used for the answer
    if not chatbot:
        for document in docs:
            print("\n> " + document[0].metadata["source"] + ":")
            print(document[0].page_content)
            print("\n> Score: " + str(document[1]))
    
    return answer['answer']