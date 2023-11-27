from langchain import HuggingFaceHub, LLMChain,PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def run_request(question_to_ask, model_type, alt_key):
    # Hugging Face model
    llm = HuggingFaceHub(huggingfacehub_api_token = alt_key, repo_id= model_type, model_kwargs={"temperature":0.1, "max_new_tokens":500})
    llm_prompt = PromptTemplate.from_template(question_to_ask)
    llm_chain = LLMChain(llm=llm,prompt=llm_prompt)
    llm_response = llm_chain.predict()
    # return the response
    llm_response = format_response(llm_response)
    return llm_response

def format_response(res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing we need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

def format_question(primer_desc,primer_code , question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
    instructions = "Create a figure object named fig using plotly express. Do not show the fig.\n"
    instructions += "Pass it to : st.plotly_chart(fig,use_container_width=True)"
    primer_desc = primer_desc.format(instructions)  
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return  '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset,df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. " 
        elif df_dataset.dtypes[i]== "datetime64[ns]":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains datetime values. " 
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    primer_desc = primer_desc + "\nAdd a title."
    primer_desc = primer_desc + "{}" # Space for additional instructions if needed
    primer_desc = primer_desc + "\nUsing Python version 3.11.5, create a script using the dataframe df to graph the following: "
    #pimer_code = "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n"
    pimer_code = "import pandas as pd\nimport plotly.express as px\nimport streamlit as st\nimport matplotlib.pyplot as plt\n"
    #pimer_code = pimer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    #pimer_code = pimer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc,pimer_code