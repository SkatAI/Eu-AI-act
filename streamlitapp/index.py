"""
TODO: filter by author
"""
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

from retrieve import Retrieve

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    st.set_page_config(
        page_title="EU AI-act Knowledge Base",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={"About": "Knowledge Base on EU AI-act"},
    )

    model_options = ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]
    # "CULT",
    # "IMCO-LIBE",
    # "ITRE",
    # "JURI",
    # "TRAN",
    # author_options = ["all versions", "2024 coreper", "2022 council", "2021 commission"]
    author_options_docs = ["all documents", "2024 coreper", "2022 council", "2021 commission"]

    author_options_amendments = [
        "all amendments",
        "ECR",
        "EPP",
        "GUE/NGL",
        "Greens/EFA",
        "ID",
        "Renew",
        "S&D",
    ]
    includes = {"recitals": True, "articles": True, "annex": True, "amendments": False}
    search_params = {
        "model": model_options[0],
        "author": author_options_docs[1],
        "includes": includes,
    }

    # ----------------------------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------------------------
    with st.sidebar:
        st.header(":orange[EU AI-Act Explorer]")

        # sections to include

        includes["recitals"] = st.checkbox(
            "Recitals",
            value=includes["recitals"]
            if st.session_state.get(f"key_incl_recitals") is None
            else st.session_state.get(f"key_incl_recitals"),
            key="key_incl_recitals",
        )
        includes["articles"] = st.checkbox(
            "Regulation",
            value=includes["articles"]
            if st.session_state.get(f"key_incl_articles") is None
            else st.session_state.get(f"key_incl_articles"),
            key="key_incl_articles",
        )
        includes["annex"] = st.checkbox(
            "annexes".capitalize(),
            value=includes["annex"]
            if st.session_state.get(f"key_incl_annex") is None
            else st.session_state.get(f"key_incl_annex"),
            key="key_incl_annex",
        )
        includes["amendments"] = st.checkbox(
            "amendments".capitalize(),
            value=includes["amendments"]
            if st.session_state.get(f"key_incl_amendments") is None
            else st.session_state.get(f"key_incl_amendments"),
            key="key_incl_amendments",
        )
        search_params.update({"includes": includes})

        # author
        if (includes["recitals"] | includes["articles"] | includes["annex"]) & includes[
            "amendments"
        ]:
            index = 1
            author_options = author_options_docs + author_options_amendments
        elif includes["amendments"]:
            index = 0
            author_options = author_options_amendments
        elif includes["recitals"] | includes["articles"] | includes["annex"]:
            index = 1
            author_options = author_options_docs
        else:
            author_options = None

        if author_options is not None:
            if st.session_state.get("author_key") is not None:
                if st.session_state.get("author_key") in author_options:
                    index = author_options.index(st.session_state.get("author_key"))

            author = st.selectbox(
                "Authored by",
                author_options,
                index=index,
                key="author_key",
                help="""
    - April 2021: The commission proposed a 1st version of the regulation in April 2021.
    - April 2022: The council then published a revised version in April 2022.
    - February 2024: The Coreper version represents the latest draft agreed after the Trilogue negocations (Dec 23).""",
            )
            search_params.update({"author": author})

        # advanced
        with st.expander("Advanced settings"):
            model = st.selectbox(
                "Generative model",
                model_options,
                index=0
                if st.session_state.get("model_key") is None
                else model_options.index(st.session_state.get("model_key")),
                key="model_key",
                help="""
    - gpt-3.5 is a faster and more concise model;
    - gpt-4 has more recent knowledge that it can use in its answers""",
            )
            search_params.update({"model": model})

            search_type_options = ["hybrid", "near_text"]
            search_type = st.selectbox(
                "Search type",
                search_type_options,
                index=0
                if st.session_state.get("search_type_key") is None
                else search_type_options.index(st.session_state.get("search_type_key")),
                key="search_type_key",
                help="""
- near_text search mode focuses on semantic proximity of the query and the retrieved paragraphs.
- hybrid search mode focuses more on important keywords; may work better for topic focused search.""",
            )
            search_params.update({"search_type": search_type})

            number_elements_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            number_elements = st.selectbox(
                "Number of retrieved elements",
                number_elements_options,
                index=4
                if st.session_state.get("number_elements_key") is None
                else number_elements_options.index(st.session_state.get("number_elements_key")),
                key="number_elements_key",
                help="""Number of retried elements used in the prompt to answer the query.""",
            )
            search_params.update({"number_elements": number_elements})
            # temperature
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                value=0.0
                if st.session_state.get("search_temperature_key") is None
                else st.session_state.get("search_temperature_key"),
                key="search_temperature_key",
                help="""Increase the temperature to generate more creative answersß""",
            )
            search_params.update({"temperature": temperature})

        st.divider()
        st.caption("[github: SkatAI/dmi2024-ai-act](https://github.com/SkatAI/Eu-AI-act)")
        st.caption("by [Université Gustave Eiffel](https://www.univ-gustave-eiffel.fr/en/)")
        # st.write(search_params)
    # ----------------------------------------------------------------------------
    # Main query input
    # ----------------------------------------------------------------------------

    st.header("[eu-ai-act.streamlit.app](https://eu-ai-act.streamlit.app)")

    with st.form("search_form", clear_on_submit=False):
        search_query = st.text_area(
            "Your query:",
            key="query_input",
            height=20,
            help="""Write a query, a question about the AI-act.""",
        )

        sc3, sc4 = st.columns([10, 1])
        with sc3:
            search_scope = st.checkbox(
                "Also generate an answer without context",
                help="""Check to generate the answer without any form of retrieval""",
            )
        with sc4:
            search_button = st.form_submit_button(label="Ask")

    # ----------------------------------------------------------------------------
    # Search results
    # ----------------------------------------------------------------------------
    if search_button:
        retr = Retrieve(search_query, search_params)
        retr.search()

        st.header(":blue[Answer:]")
        retr.generate_answer_with_context()
        _, col2 = st.columns([1, 15])
        with col2:
            st.markdown(f"{retr.answer_with_context}", unsafe_allow_html=True)
        st.divider()

        if search_scope:
            retr.generate_answer_bare()
            st.subheader(":magenta[Answer without context:]")
            _, col2 = st.columns([1, 15])
            with col2:
                st.markdown(f"{retr.answer_bare}", unsafe_allow_html=True)

        st.subheader(":gray[Retrieved documents]")

        for i in range(len(retr.response.objects)):
            try:
                title = " - ".join(retr.retrieved_title(i).split("-")[-2:]).replace("**", "")
            except:
                title = retr.retrieved_title(i)

            with st.expander(f"{i+1}) :gray[{title}]"):
                retr.format_properties(i)
                retr.format_metadata(i)

        retr.log_session()
        # conn = st.connection("postgresql", type="sql")
        # retr.to_db(conn)

        # retr.client_close()
