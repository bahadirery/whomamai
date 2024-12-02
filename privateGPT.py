#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
import os
import argparse
import time
import statistics
import re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, DirectoryLoader, BSHTMLLoader
from typing import List
from langchain_core.prompts import ChatPromptTemplate
import shutil

from pathlib import Path

load_dotenv()


embeddings_model_name = 'all-MiniLM-L12-v2'
persist_directory = 'chroma_kamy/'
persist_directory_filtered = 'chroma_kamy_filtered/'
source_docs_path = 'wispermed_party/source_data/individual_texts'
model_type = "mistralai/Mistral-7B-Instruct-v0.2"


# model_path = os.environ.get('MODEL_PATH')
# model_n_ctx = os.environ.get('MODEL_N_CTX')
# model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
# target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    #db_filtered = Chroma(collection_name="wispermed_party_filtered", embedding_function = embeddings, persist_directory=persist_directory_filtered)
    #retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=True)
    else:
        print(f"Model {model_type} not supported!")
        print("Try to load the model from huggingface")
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_type,
            task="text-generation",
            model_kwargs={"temperature": 0.7, "do_sample": True, "max_length": 10000, "trust_remote_code": True, "cache_dir": "/local/work/baheryilmaz/.cache"},
            device=0
        )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        # Get the answer from the chain
        start = time.time()
        #docs = retriever.get_relevant_documents(query)
        pattern = r"(id)(\W)*([0-9]+)"
        # Define the UUID pattern
        uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
        filter = {}
        for match in re.finditer(uuid_pattern, query):
            uuid_id = match.group(0)
            filter = {'source': f'{source_docs_path}/{uuid_id}.txt'}
        docs_with_score = db.similarity_search_with_score(query,filter=filter)
                                                          
        #ec2a5c45-c8c5-4086-9799-bf16bd12ee0d
        retriever = db.as_retriever(filter=filter, search_kwargs={"k": 8})
        #
        # Always returns that one person
        #person_metadata = docs_with_score[0][0].metadata
        #person_page_content = docs_with_score[0][0].page_content
        #mean = statistics.mean([doc[1] for doc in docs_with_score])
        docs = [doc for doc in docs_with_score ]
        doc_page_contens = [doc[0] for doc in docs_with_score]
        #memory = ConversationBufferMemory(memory_key="text")
        
        ids = [str(i) for i in range(1, len(docs) + 1)]
        db_filtered = Chroma.from_documents(doc_page_contens, embeddings, ids=ids) # We do this because we dont want to accumulate same filtered documents
        #db_filtered.add_documents(doc_page_contens)
        retriever_filtered = db_filtered.as_retriever(search_kwargs={"k": 100})
        
    
        prompt_template = """You are an assistant to retrieve information from set of documents that belongs to individuals.
                            In the documents there are questions and answers separated with "|", create a final answer to a user question  with references ("SOURCES").
                            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
                            ALWAYS return a "SOURCES" part in your answer.
                                    
        USER QUESTION: {question}
        =========
        {source_documents}
        =========
        FINAL ANSWER:"""
        PROMPT = ChatPromptTemplate.from_template(prompt_template)
        
        if len(doc_page_contens) > 0:
            retriever_final = retriever_filtered
        else:
            retriever_final = retriever
        
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

        #answer = chain({"input_documents": doc_page_contens, "human_input": query}, return_only_outputs=True)
        #answer = chain.run({"input_documents": doc_page_contens, "human_input": query})
        answer = rag_chain.invoke(query)
        #res = qa(query)
        #answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        #reset the db
        #shutil.rmtree(persist_directory_filtered)
        #db_filtered._collection.delete(ids=[ids[-1]])
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer['answer'])

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document[0].metadata["source"] + ":")
            print(document[0].page_content)
            print("\n> Score: " + str(document[1]))
        #print(chain.memory.buffer)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
