import streamlit as st
import pandas as pd
from helper_functions import generate_rag, load_vector_store, auto_scroll_to_bottom
import warnings
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub,PromptTemplate

st.set_page_config(
    page_title="Build a data visualisation bot, powered by Code-Llama and Zephyr-Beta",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("Creating Visualisations using Natural Language with Code Llama 💬")

available_models = {"Code Llama":"codellama/CodeLlama-34b-Instruct-hf", 
                    "Zephyr Beta":"HuggingFaceH4/zephyr-7b-beta"
                    }

rag_model = "Zephyr Beta"    
hf_key = st.secrets["hf_key"]

if "datasets" not in st.session_state:
    datasets = {}
    #add local datasets here
    st.session_state["datasets"] = datasets 
else:
    datasets = st.session_state["datasets"]

with st.sidebar:
    dataset_container = st.empty()

    try:
        uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
        index_no=0
        if uploaded_file:
            file_name = uploaded_file.name[:-4].capitalize()
            datasets[file_name] = pd.read_csv(uploaded_file)
            index_no = len(datasets)-1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:",datasets.keys(),index=index_no)

if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello ! How can I help you ?"},
    ]

if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    if "figure" in message:
        with st.chat_message(message["role"]):
            st.plotly_chart(message["figure"], use_container_width=True)
            st.write(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

vectordb = load_vector_store()

template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try 
to make up an answer. 
Keep the answer as concise as possible. Use 1 sentence to sum all points up.
______________
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = HuggingFaceHub(huggingfacehub_api_token = hf_key, repo_id= available_models[rag_model], model_kwargs={"temperature":0.01, "max_new_tokens":500})

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rag_answer = qa_chain({"query": prompt})

            print(rag_answer)
            answer = generate_rag(rag_answer, prompt, available_models[rag_model], alt_key=hf_key)
            st.write(answer)

            st.session_state.messages.append(answer)

auto_scroll_to_bottom()

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
