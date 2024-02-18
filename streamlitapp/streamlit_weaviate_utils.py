import os, re, json
import time, datetime
import pandas as pd
import numpy as np

import weaviate
import weaviate.classes as wvc


def count_collection(collection):
    return collection.aggregate.over_all(total_count=True).total_count

def list_collections(client):
    for collection_name in client.collections.list_all().keys():
        props = [(p.name, p.vectorizer, p.vectorizer_config.skip ) for p in  client.collections.list_all()[collection_name].properties]
        collection = client.collections.get(collection_name)
        count_ = collection.aggregate.over_all(total_count=True).total_count
        print(f"* {collection_name} [{count_}]: \n\t {props}")


def which_text_splitter(cfg):
    # ------------------------------------
    # define text splitter
    # ------------------------------------
    if cfg["splitter"] == "tiktoken":
        from langchain.text_splitter import CharacterTextSplitter
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=cfg["chunk_size"],
            chunk_overlap = cfg["chunk_overlap"],
            separator = cfg["separator"],
        )
    elif cfg["splitter"] == "nltk":
        from langchain.text_splitter import NLTKTextSplitter
        text_splitter = NLTKTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separator = cfg["separator"]
        )
    elif cfg["splitter"] == "huggingface_tokenizer":
        from langchain.text_splitter import CharacterTextSplitter
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=500, chunk_overlap=50
        )
    return text_splitter


def which_vectorizer(vectorizer_name):
    if vectorizer_name == "OpenAI":
        vectorizer  = wvc.Configure.Vectorizer.text2vec_openai(vectorize_collection_name = False)
    elif vectorizer_name == "Camembert":
        vectorizer  = wvc.Configure.Vectorizer.text2vec_huggingface(
                model= "dangvantuan/sentence-camembert-large",
                wait_for_model = True,
                endpoint_url = "https://ulbcpws10z5vej7w.eu-west-1.aws.endpoints.huggingface.cloud",
                vectorize_class_name = False,
            )
    return vectorizer

def connect_client(location = 'local'):
    assert location in ['local', 'cloud'], f"location: {location} is not local or cloud"
    if location == 'local':
        # connect to weaviate client (on local)
        print("connection to local")
        client = weaviate.connect_to_local(
                port=8080,
                grpc_port=50051,
                headers={
                    "X-OpenAI-Api-Key": os.environ["OPENAI_APIKEY"],
                }
            )

    if location == 'cloud':
        print("connection to wcs")
        client = weaviate.connect_to_wcs(
                    cluster_url=os.environ["WEAVIATE_CLUSTER_URL"],
                    auth_credentials=weaviate.AuthApiKey(os.environ["WEAVIATE_KEY"]),
                    headers={
                        "X-OpenAI-Api-Key": os.environ["OPENAI_APIKEY"],
                    }
            )
    return client

def create_collection(client, collection_name, vectorizer, properties, replace = True, stopwords = None ):
    # create collection
    if client.collections.exists(collection_name):
        print("-- collection exists ")
        if replace :
            print("-- deleting collection")
            client.collections.delete(collection_name)

    if not client.collections.exists(collection_name):
        print("-- creating collection")
        collection = client.collections.create(
            name= collection_name,
            vectorizer_config=vectorizer,
            properties=properties
        )

        if stopwords is not None:
            print(f"-- adding {len(stopwords)} stopwords ")
            collection.config.update(
                inverted_index_config=wvc.Reconfigure.inverted_index(
                    stopwords_additions = stopwords
                )
            )
    # reload collection
    return client.collections.get(collection_name)

def store_embed(items, collection):
    # insert and vectorize
    batch_result = collection.data.insert_many(items)
    if batch_result.has_errors:
        print(batch_result.errors)
        raise "stopping"
