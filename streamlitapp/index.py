'''
TODO: filter by author
'''
import os, re, json, glob
import time, datetime
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
from streamlit_weaviate_utils import *
from retrieve import Retrieve

import yaml
from yaml.loader import SafeLoader

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    st.set_page_config( page_title="EU AI-act Knowledge Base", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items={"About": "Knowledge Base on EU AI-act"})

    model_options = ["gpt-3.5-turbo-1106","gpt-4-1106-preview"]
    author_options = ['--','COMMISSION', 'COUNCIL',  'PARLIAMENT', 'CULT', 'ECR', 'EPP', 'IMCO-LIBE', 'ITRE', 'JURI', 'TRAN', 'GUE/NGL','Greens/EFA', 'ID', 'Renew', 'S&D']
    author_options = ['all versions', '2024 coreper', '2022 council', '2021 commission']

    search_params = {
        'model' : model_options[0],
        'author' : author_options[1],
    }

    # ----------------------------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------------------------
    with st.sidebar:
        st.header(":orange[EU AI-Act Explorer]")

        model = st.selectbox(
                "Generative model",
                model_options,
                index = 0 if st.session_state.get("model_key") is None else model_options.index(st.session_state.get("model_key")),
                key = "model_key",
                help = '''
- gpt-3.5 is a faster and more concise model;
- gpt-4 has more recent knowledge that it can use in its answers''')

        search_params.update({'model': model})
        # author
        # TODO: change to commission, council, coreper
        author = st.selectbox("Authored by", author_options,
                index = 3 if st.session_state.get("author_key") is None else author_options.index(st.session_state.get("author_key")),
                key = "author_key",
                help = '''
- April 2021: The commission proposed a 1st version of the regulation in April 2021.
- April 2022: The council then published a revised version in April 2022.
- February 2024: The Coreper version represents the latest draft agreed after the Trilogue negocations (Dec 23).''')
        search_params.update({'author': author})

        # advanced
        with st.expander("Advanced settings"):
            search_type_options = ["hybrid", "near_text"]
            search_type = st.selectbox("Search type", search_type_options,
                    index = 0 if st.session_state.get("search_type_key") is None else search_type_options.index(st.session_state.get("search_type_key")),
                    key = "search_type_key",
                    help='''
- near_text search mode focuses on semantic proximity of the query and the retrieved paragraphs.
- hybrid search mode focuses more on important keywords; may work better for topic focused search.''')
            search_params.update({'search_type': search_type})


            number_elements_options = [1,2,3,4,5,6,7,8,9,10]
            number_elements = st.selectbox("Number of retrieved elements", number_elements_options,
                    index = 4 if st.session_state.get("number_elements_key") is None else number_elements_options.index(st.session_state.get("number_elements_key")),
                    key = "number_elements_key",
                    help='''Number of retried elements used in the prompt to answer the query.'''
            )
            search_params.update({'number_elements': number_elements})
            # temperature
            temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, step=0.1,
                    value= 0.0 if st.session_state.get("search_temperature_key") is None else st.session_state.get("search_temperature_key"),
                    key = "search_temperature_key",
                    help='''Increase the temperature to generate more creative answersß'''
            )
            search_params.update({'temperature': temperature})

        st.divider()
        st.caption("[github: SkatAI/dmi2024-ai-act](https://github.com/SkatAI/dmi2024-ai-act)")
        st.caption("by [Université Gustave Eiffel](https://www.univ-gustave-eiffel.fr/en/)")
        st.write(search_params)
    # ----------------------------------------------------------------------------
    # Main query input
    # ----------------------------------------------------------------------------

    search_col1, search_c2 = st.columns([8,4])
    with search_col1:
        with st.form('search_form', clear_on_submit = False):
            search_query = st.text_area(
                "Your query:",
                key="query_input",
                height = 20,
                help = '''Write a query, a question about the AI-act.'''
                )

            sc3, sc4 = st.columns([10,1])
            with sc3:
                search_scope = st.checkbox("Show the rough answer without context", help='''Check to generate the answer without any form of retrieval''')
            with sc4:
                search_button = st.form_submit_button(label="Ask")

    # ----------------------------------------------------------------------------
    # Search results
    # ----------------------------------------------------------------------------
    if search_button:
        # TODO: nasty call, dict much?
        retr = Retrieve(search_query, search_params)
        # retr = Retrieve(search_query, search_type, model, number_elements, temperature, author)
        retr.search()

        if search_scope:
            st.subheader("Answer without context:")
            retr.generate_answer_bare()
            _, col2 = st.columns([1, 15])
            with col2:
                st.markdown(f"<em>{retr.answer_bare}</em>", unsafe_allow_html=True)

        st.subheader("Answer with retrieval")
        retr.generate_answer_with_context()
        _, col2 = st.columns([1, 15])
        with col2:
            st.markdown(f"<em>{retr.answer_with_context}</em>", unsafe_allow_html=True)

        retr.save()
        st.header("Retrieved documents")

        for i in range(len(retr.response.objects)):
            with st.expander(retr.retrieved_title(i)):
                col1, col2 = st.columns([1, 15])
                with col1:
                    st.subheader(f"{i+1})")
                with col2:
                    retr.format_properties(i)
                    retr.format_metadata(i)
                    st.divider()
