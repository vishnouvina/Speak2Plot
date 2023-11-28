import streamlit as st
import pandas as pd

from classes import get_primer, format_question, run_request
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

def auto_scroll_to_bottom():
    st.markdown(
        '<script>window.scrollTo(0,document.body.scrollHeight);</script>', 
        unsafe_allow_html=True
    )

st.header("Creating Visualisations using Natural Language with Code Llama 💬")

available_models = {"Code Llama":"codellama/CodeLlama-34b-Instruct-hf", 
                    #"Zephyr Beta":"HuggingFaceH4/zephyr-7b-beta"
                    }

hf_key = st.secrets["hf_key"]

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


if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "What do you want to see ?"}
    ]

if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

primer1, primer2 = get_primer(datasets[chosen_dataset],'datasets["'+ chosen_dataset + '"]') 

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                question_to_ask = format_question(primer1, primer2, prompt, selected_model)   
                answer=""
                answer = run_request(question_to_ask, available_models[selected_model], alt_key=hf_key)
                answer = primer2 + answer
                print("Model: " + selected_model)
                print(answer)
                exec(answer)
                message = {"role": "assistant", "content": fig}
            
            except Exception as e:
                print(e)
                message = {"role": "assistant", "content": "Unfortunately the code generated from the model contained errors and was unable to execute."}
                st.write(message['content'])
            st.session_state.messages.append(message) 

auto_scroll_to_bottom()

def tabs_datasets():
    tab_list = st.tabs(datasets.keys())

    # Load up each tab with a dataset
    for dataset_num, tab in enumerate(tab_list):
        with tab:
            # Can't get the name of the tab! Can't index key list. So convert to list and index
            dataset_name = list(datasets.keys())[dataset_num]
            st.subheader(dataset_name)
            st.dataframe(datasets[dataset_name],hide_index=True)

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)