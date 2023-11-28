import pandas as pd
import streamlit as st

from classes import get_primer, format_question, run_request
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide",page_title="Chat2VIS")

st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Creating Visualisations using Natural Language with Code Llama</h2>", unsafe_allow_html=True)

available_models = {"Code Llama":"codellama/CodeLlama-34b-Instruct-hf", 
                    "Zephyr Beta":"HuggingFaceH4/zephyr-7b-beta"
                    }

hf_key = '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'

# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    #datasets["Feedbacks"] = clean_dataset(pd.read_csv("data/translated_feedbacks.csv"), available_models["Code Llama"], hf_key)
    #datasets["History"] = clean_dataset(pd.read_csv("data/history.csv"), available_models["Code Llama"], hf_key)
    datasets["Sleep"] = pd.read_csv("data/physionet_cleaned/sleep.csv", parse_dates=['date'])
    datasets["Screen"] = pd.read_csv("data/physionet_cleaned/screen.csv", parse_dates=['date'])
    datasets["Steps"] = pd.read_csv("data/physionet_cleaned/steps.csv", parse_dates=['date'])
    st.session_state["datasets"] = datasets 
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]

with st.sidebar:
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()

    # Add facility to upload a dataset
    try:
        uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
        index_no=0
        if uploaded_file:
            # Read in the data, add it to the list of available datasets. Give it a nice name.
            file_name = uploaded_file.name[:-4].capitalize()
            # Clean the chosen dataset
            datasets[file_name] = pd.read_csv(uploaded_file)
            # We want to default the radio button to the newly added dataset
            index_no = len(datasets)-1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:",datasets.keys(),index=index_no)#,horizontal=True,)

    selected_model = st.radio(
    ":brain: Choose your model:", available_models.keys(),
    captions = available_models.values())

 # Text area for query
question = st.text_area("What would you like to visualise?",height=10)
go_btn = st.button("Go...")

# Make a list of the models which have been selected
selected_models = [selected_model]
model_count = len(selected_models)

# Execute chatbot query
if go_btn and model_count > 0:
    # Place for plots depending on how many models
    #plots = st.tabs(model_count)
    # Get the primer for this dataset
    primer1,primer2 = get_primer(datasets[chosen_dataset],'datasets["'+ chosen_dataset + '"]') 
    # Create model, run the request and print the results
    for plot_num, model_type in enumerate(selected_models):
        #with plots[plot_num]:
        with st.container():
            st.subheader(model_type)
            try:
                # Format the question 
                question_to_ask = format_question(primer1, primer2, question, model_type)   
                # Run the question
                answer=""
                answer = run_request(question_to_ask, available_models[model_type], alt_key=hf_key)
                # the answer is the completed Python script so add to the beginning of the script to it.
                answer = primer2 + answer
                print("Model: " + model_type)
                print(answer)
                #plot_area = st.empty()
                #plot_area.pyplot(exec(answer))
                exec(answer)           
            except Exception as e:
                print(e)
                st.error("Unfortunately the code generated from the model contained errors and was unable to execute.")

# Display the datasets in a list of tabs
# Create the tabs
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