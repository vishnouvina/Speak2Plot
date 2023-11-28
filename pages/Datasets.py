import streamlit as st
import pandas as pd
import warnings

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
                    #"Zephyr Beta":"HuggingFaceH4/zephyr-7b-beta"
                    }
    
if "datasets" not in st.session_state:
    datasets = {}
    datasets["Sleep"] = pd.read_csv("data/physionet_cleaned/sleep.csv", parse_dates=['date'])
    datasets["Screen"] = pd.read_csv("data/physionet_cleaned/screen.csv", parse_dates=['date'])
    datasets["Steps"] = pd.read_csv("data/physionet_cleaned/steps.csv", parse_dates=['date'])
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

    selected_model = st.radio(
    ":brain: Choose your model:", available_models.keys(),
    captions = available_models.values())

tab_list = st.tabs(datasets.keys())

for dataset_num, tab in enumerate(tab_list):
    with tab:
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name],hide_index=True)