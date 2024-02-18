import os, re, json, glob
import pandas as pd
import numpy as np

# streamlit
import streamlit as st

# weaviate
from weaviate.classes import Filter

# open AI
from openai import OpenAI

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

# local
from  streamlit_weaviate_utils import *

class Retrieve(object):

    # def __init__(self, query, search_type, model, number_elements, temperature, author):
    def __init__(self, query, search_params):
        # collection_name = "AIActKnowledgeBase"
        self.authors = {'all versions': None,
            '2024 coreper':'coreper', '2022 council':'council', '2021 commission': 'commission'}
        collection_name = "AIAct_240218"
        cluster_location = "local"
        self.client = connect_client(cluster_location)
        assert self.client is not None
        assert self.client.is_live()

        # retrieval
        self.vectorizer = which_vectorizer("OpenAI")
        self.collection = self.client.collections.get(collection_name)
        # TODO safeguard query
        self.query = query
        self.author = self.authors.get(search_params.get('author'))
        self.model = search_params.get('model')
        self.search_type = search_params.get('search_type')
        self.response_count_ = search_params.get('number_elements')
        self.temperature = search_params.get('temperature')
        print(self.__dict__)
        # generative

        self.prompt_generative_context =ChatPromptTemplate.from_template(
'''You are a journalist from Euractiv who is an expert on both Artifical Intelligence, the AI-act regulation from the European Union and European policy making.
Your goal is to make it easier for people to understand the AI-Act from the UE.

You are given a query and context information.
Your specific task is to answer the query based on the context information.

# make sure:
- If the context does not provide an answer to the query, use your global knowledge to write an answer.
- If the context is not related to the query, clearly state that the context does not provide information with regard to the query.
- Your answer must be short and concise.
- Important: if you don't have the necessary information, say so clearly. Do not try to imagine an answer

--- Context:
{context}
--- Query:
{query}''')

        self.prompt_generative_bare = ChatPromptTemplate.from_template(
'''You are a journalist from Euractiv who is an expert on both Artifical Intelligence, the AI-act regulation from the European Union and European policy making.
Your goal is to make it easier for people to understand the AI-Act from the UE.

# make sure to:
- clearly state if you can't find the answer to the query. Do not try to invent an answer.
- Focus on the differences in the text between the European union entities: commission, council, parliament as well as the political groups and committees.
- Your answer must be short and concise. One line if possible
- Do not try to imagine a fake answer if you don't have the necessary information

--- Query:
{query}''')



        self.llm = ChatOpenAI(temperature=self.temperature, model=self.model)
        self.context_chain   = LLMChain(llm=self.llm, prompt=self.prompt_generative_context,  output_key="answer_context", verbose=False)

        self.overall_context_chain = SequentialChain(
            chains=[self.context_chain],
            input_variables=["context", "query"],
            output_variables=["answer_context"],
            verbose=True
        )

        self.bare_chain   = LLMChain(llm=self.llm, prompt=self.prompt_generative_bare,  output_key="answer_bare", verbose=False)

        self.overall_bare_chain = SequentialChain(
            chains=[self.bare_chain],
            input_variables=["query"],
            output_variables=["answer_bare"],
            verbose=True
        )

    def search(self):
        filters = None
        if self.author is not None:
            filters = Filter("author").equal(self.author)

        if self.search_type == "hybrid":
            self.response = self.collection.query.hybrid(
                    query=self.query,
                    query_properties = ['text'],
                    filters= filters,
                    limit=self.response_count_,
                    return_metadata = ['score', 'explain_score', 'is_consistent'],
                )
        elif self.search_type == "near_text":
            self.response = self.collection.query.near_text(
                    query= self.query,
                    filters= filters,
                    limit=self.response_count_,
                    return_metadata = ['distance','certainty'],
                )
        self.get_context()

    def generate_answer_with_context(self):
        self.response_context = self.overall_context_chain({ "context": self.context, "query": self.query })
        self.answer_with_context = self.response_context['answer_context']

    def generate_answer_bare(self):
        self.response_bare = self.overall_bare_chain({ "query": self.query })
        self.answer_bare = self.response_bare['answer_bare']

    def get_context(self):
        texts = []
        for i in range(self.response_count_):
            prop = self.response.objects[i].properties
            text = "---"
            if prop.get('section') == 'recitals':
                location = prop.get('rec')
            if prop.get('section') == 'articles':
                location = [prop.get(lct) for lct in ['ttl', 'art', 'par'] if prop.get(lct) is not None ]
                location = ">> ".join(location)

            text += ' - '.join([location, prop.get('author') ])
            text += "\n"
            text += prop.get('text')

            texts.append(text)
        self.context = '\n'.join(texts)

    # format
    def format_metadata(self, i):
        metadata_str =[]
        if self.search_type == "hybrid":
            metadata_str =f"**score**: {np.round(self.response.objects[i].metadata.score, 4)} "
        elif self.search_type == "near_text":
            metadata_str = f"distance: {np.round(self.response.objects[i].metadata.distance, 4)} certainty: {np.round(self.response.objects[i].metadata.certainty, 4)} "
        st.write(metadata_str)

    def retrieved_title(self, i):
        # TODO: duplicated code with get_context
        prop = self.response.objects[i].properties
        # headlinize
        # title = prop['text'].split('\n')[0].strip()
        # title = ' - '.join([prop['title'], prop['numbering'], prop['author'] ])
        print('---')
        print(prop)
        print('---')
        if prop.get('section') == 'recitals':
            location = prop.get('rec')
        if prop.get('section') == 'articles':
            location = [prop.get(lct) for lct in ['ttl', 'art', 'par'] if prop.get(lct) is not None ]
            location = " >> ".join(location)

        title = ' - '.join([location, prop.get('author') ])

        return  f"**{title}**"

    def format_properties(self, i):
        prop = self.response.objects[i].properties
        # remaining_words_count = len(prop['text'].split(' ')) - 100
        # text = ' '.join(prop['text'].split(' ')[:100])
        # headlinize
        # text = prop['text'].split("\n")
        # text[0] = f"**{text[0].strip()}**"
        # text = '\n'.join(text).replace('\n', '  \n')
        st.write(prop['text'].strip())

        # url = f"  [{prop['pid']}]({prop['url']}) "
        # authors = ', '.join(prop['authors'])
        # if 'tags' in prop.keys():
        #     tags = ', '.join(prop['tags'])
        # else:
        #     tags = ''
        # st.caption(' - '.join([prop['source'], prop['source_type'], prop['text_type'], authors, url, tags]))

    def save(self):
        # print(self.__dict__.keys())
        os.write(1,bytes("--"*20 + "\n", 'utf-8'))
        os.write(1,bytes(f"query: {self.query}\n" , 'utf-8'))
        os.write(1,bytes(f"search_type: {self.search_type}\n" , 'utf-8'))
        os.write(1,bytes(f"model: {self.model}\n" , 'utf-8'))
        os.write(1,bytes(f"response_count_: {self.response_count_}\n" , 'utf-8'))
        os.write(1,bytes(f"temperature: {self.temperature}\n" , 'utf-8'))
        os.write(1,bytes(f"author: {self.author}\n" , 'utf-8'))
        os.write(1,bytes(f"answer with context:\n {self.answer_with_context}\n" , 'utf-8'))
        os.write(1,bytes("--"*20 + "\n\n", 'utf-8'))
        pass


def perform_search(query, search_type, model):
    '''
    - vectorize query
        - client, vectorizer etc
        - store query
    - near text search
    - prompt
    - answer generation
    '''
    retr = Retrieve(query, search_type, model)
    retr.search()

    return retr


