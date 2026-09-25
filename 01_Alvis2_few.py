from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage 
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langsmith.schemas import Run, Example
import pandas as pd
import ast
import os
from dotenv import load_dotenv, dotenv_values
import random
import json
import time 





Prompt = "FEW"
reasoning_effort = "none"
task = "diagnosis"
web_search = "no"


def load_keys():
    
    load_dotenv() 
    KEYS_LIST = []
    env_vars = dotenv_values(".env")
    n = 1
    while n <= len(env_vars):
        KEYS_LIST.append(os.getenv(f"LLM_API_KEY_{n}"))
        n+=1
    return KEYS_LIST


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    



obtained_files = []

@tool
def nodule_detector(CT_name: str) -> str:
    """
    Scans a CT image to detect potential pulmonary nodules.
    Args: CT_name: The filename of the CT scan to analyze.
    """
    nodule_id = f"NODULE_{random.randint(100, 999)}"
    result = f"Suspicious mass '{nodule_id}' in file {CT_name}."
    obtained_files.append(result)
    return result

@tool
def nodule_segmentator(CT_name: str, nodule_id: str) -> str:
    """
    Creates a binary mask isolating a specific nodule from the surrounding lung tissue    
    Args: A valid filename of the CT scan to analyze together with the nodule id
    """
    segmented_filename = f"{CT_name}_SGM"
    obtained_files.append(segmented_filename)
    return segmented_filename

@tool
def nodule_classifier(CT_name_sgm: str) -> str:
    """
    Analyzes the morphology and texture of a segmented nodule to classify it.
    Args: a segmented_nodule_name: The filename of the segmented nodule.
    """
    if CT_name_sgm[:2] in ("15","18","21","24"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_1_CENTRAL_MASS"

    elif CT_name_sgm[:2] in ("16", "19", "22", "25"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_1_PERIPHERAL_MASS"

    elif CT_name_sgm[:2] in ("17", "20", "23", "26"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_1_METASTASIS"

    elif CT_name_sgm[:2] in ("27", "30", "33", "36"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_2_CENTRAL_MASS"

    elif CT_name_sgm[:2] in ("28", "31", "34", "37"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_2_PERIPHERAL_MASS"

    elif CT_name_sgm[:2] in ("29", "32", "35", "38"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_2_METASTASIS"

    elif CT_name_sgm[:2] in ("39","42","45","48"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_3_CENTRAL_MASS"

    elif CT_name_sgm[:2] in ("40", "43", "46", "49"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_3_PERIPHERAL_MASS"

    elif CT_name_sgm[:2] in ("41", "44", "47", "50"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_3_METASTASIS"

    elif CT_name_sgm[:2] in ("51","54","57","60"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_4_CENTRAL_MASS"

    elif CT_name_sgm[:2] in ("52", "55", "58", "61"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_4_PERIPHERAL_MASS"

    elif CT_name_sgm[:2] in ("53", "56","59","62"):
        base_name = CT_name_sgm[:-4]
        CT_name_class = base_name + "_STAGE_4_METASTASIS"

    else:
        CT_name_class = "wrong filename, retry" 

    obtained_files.append(CT_name_class)
    return CT_name_class

@tool
def cell_detector(biopsy_slide_name: str) -> str:
    """
    Analyzes a biopsy slide (tissue slide) to detect the coordinates of individual cells (nuclei).
    You must use this tool if a slide file is provided.
    DO NOT use this tool on radiological images (CT, X-Ray).
    Args: the slide file name
    """
    result = f"Cells detected in {biopsy_slide_name}:"
    obtained_files.append(result)
    return result

@tool
def cell_segmentator(biopsy_slide_name: str) -> str:
    """
    Performs semantic segmentation on a Slide file to outline cellular structures.
    
    Args: the slide file name 
    """
    segmentation = biopsy_slide_name + "_SGM"
    obtained_files.append(segmentation)
    return segmentation

@tool
def cell_classifier(sgm_biopsy: str) -> str:
    """
    Classifies individual cells within a segmented slide into biological categories.
    
    Args: A slide already segmented
    """
    if sgm_biopsy[:2] in ("15","16","17","27","28","29","39","40","41","51","52","53"):
        base_name = sgm_biopsy[:-4]
        class_slide = base_name + "_NEGATIVE"

    elif sgm_biopsy[:2] in ("18","19","20","30","31","32","42","43","44","54","55","56"):
        base_name = sgm_biopsy[:-4]
        class_slide = base_name + "_UNKNOWN_CANCER"

    elif sgm_biopsy[:2] in ("21","22","23","33","34","35","45","46","47","57","58","59"):
        base_name = sgm_biopsy[:-4]
        class_slide = base_name + "_ADENOCARCINOMA"

    elif sgm_biopsy[:2] in ("24","25","26","36","37","38","48","49","50","60","61","62"):
        base_name = sgm_biopsy[:-4]
        class_slide = base_name + "_SQUAMOUS_CARCINOMA"
    
    obtained_files.append(class_slide)
    return class_slide

@tool
def radiology_report_generator(CT: str, PET:str) -> str:
    """
    Generates a textual report from a CT_SCAN and/or a PET already classified (you dont need both but its better if have them both).
    Args: a valid CT_SCAN and/or a PET already classified.
    """
    if "SYNTHETIC" in CT:
        CT = CT.replace("SYNTHETIC", "") 

    if "CT_" in CT:
        report = CT.removeprefix("CT_") + "_RADIOLOGY_REPORT"
    


    obtained_files.append(report)
    return report

@tool
def biopsy_report_generator(biopsy_slide_class: str) -> str:
    """
    Generates a final histopathological report based on the microscopic analysis of a biopsy slide already classified.
    Args: tissue_slide_name_class (str): The filename of the digital biopsy slide THAT HAS BEEN ALREADY classified to analyze.
    """
    biopsy_report = biopsy_slide_class[:-6] + "_PATHOLOGY_REPORT"
    obtained_files.append(biopsy_report)
    return biopsy_report

@tool
def domain_translator(exam_name: str) -> str:
    """
    Performs cross-modality synthesis to translate medical images between domains CT to PET or PET to CT.
    Always use this tool at first if a CT or a PET is available (DO NOT USE if BOTH are available).
    
    - If input is a CT scan (starts with 'CT_'), it generates a synthetic PET scan (metabolic view).
    - If input is a PET scan (starts with 'PET_'), it generates a synthetic CT scan (anatomical view).

    Args: image_name (str): The filename of the current scan (e.g., 'CT_PatientA' or 'PET_PatientB').
    """
    if "CT_" in exam_name:
        new_name = exam_name.replace("CT_", "PET_") + "_SYNTHETIC" 
    elif "PET_" in exam_name:
        new_name =  exam_name.replace("PET_", "CT_") + "_SYNTHETIC"

        
    obtained_files.append(new_name)
    return new_name

@tool
def mutation_predictor(biopsy_slide: str) -> str:
    """
    Use this tool only if a slide file is available.
    This tool highlights the biomarkes given a biopsy slide. 
    Args: a valid tissue slide.
    """
    biomarkers = "BIOMARKERS_" + biopsy_slide.removeprefix("BIOPSY_SLIDE_") 
    obtained_files.append(biomarkers)
    return biomarkers



d_tools = [mutation_predictor, biopsy_report_generator, nodule_detector, nodule_segmentator, nodule_classifier, radiology_report_generator, domain_translator, cell_segmentator, cell_classifier, cell_detector]

def create_model(key):
    return ChatOpenAI(
        model="deepseek-v3.2:cloud",
        openai_api_key=key,
        base_url="https://ollama.com/v1",
        temperature=0.1
    ).bind_tools(d_tools)

 
def Diagnosis_Agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
            
        """
           Act as a specialized medical assistant dedicated to lung cancer diagnosis. Your goal is to help the user through the diagnosis procedure. 
           The user will provide some files that you will need to tool call. 
           Here are some rules of the diagnostic procedure:
           If the user provides either a CT or a PET scan (but not both), execute the domain_translator as the first step.
           Never execute a tool to generate an exam or report if the user has already provided it (e.g., do not call the pathology report generator if a pathology report is provided).
           Genrate radiology report only after the ct has been correctly classified.
           If biopsy is available call the mutation predictor tool before the cell detector tool. If you are going to use the biopsy report tool, make sure that you have already obtained (or the user has provided one) a cell classification on the slide. 
           Final Output: Once tool execution is complete, output your final consideration to the clinician as your last message with the follow-up management based on guidelines.
           Example 1
           This is an example query: “Hi, i have this files available: 15_PET_MARIO_ROSSI, 15_TISSUE_SLIDE_MARIO_ROSSI, could you help me with the diagnosis?”
           And these are the correct tools in the correct order that you should call relative to this example query: ['domain_translator', 'nodule_detector', 'nodule_segmentator', 'nodule_classifier', 'radiology_report_generator', 'mutation_predictor', 'cell_detector', 'cell_segmentator', 'cell_classifier', 'biopsy_report_generator'].
           Example 2
           "This is an example query: Hi, i have this files available: 14_PET_MARIO_ROSSI, could you help me with the diagnosis?"      
           And these are the correct tools in the correct order that you should call relative to this example query: ['domain_translator', 'nodule_detector','nodule_segmentator','nodule_classifier', 'radiology_report_generator']
           Example 3
           "This is an example query: Hi, i have this files available: 15_CT_MARIO_ROSSI, 15_RADIOLOGY_REPORT_MARIO_ROSSI_STAGE_1_CENTRAL_MASS, could you help me with the diagnosis?" 
           And these are the correct tools in the correct order that you should call relative to this example query: ['domain_translator', 'nodule_detector','nodule_segmentator','nodule_classifier']
           Style: No emojis, no summaries of your thinking process, and no bullet points. Be concise and use proper medical language, as you are speaking to a clinician.    
        
        """
       )
    response = d_model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):  
    messages = state["messages"]         
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"


d_graph = StateGraph(AgentState)
d_graph.add_node("Diagnosis_Agent", Diagnosis_Agent)
tool_node = ToolNode(tools=d_tools)
d_graph.add_node("tools", tool_node)
d_graph.set_entry_point("Diagnosis_Agent")
d_graph.add_conditional_edges(
    "Diagnosis_Agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
d_graph.add_edge("tools", "Diagnosis_Agent")
diagnosis = d_graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def extract_tool_calls(final_state):
    
    tool_used = []
    for message in final_state['messages']:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tc in message.tool_calls:
                tool_used.append(tc['name'])
    return tool_used



def load_data():

    data = pd.read_excel("diagnosis_dataset.xlsx")
    dataset = {}

    for i, (_, row) in enumerate(data.iterrows(), start=1): 
        files = ""

        ID = str(row.iloc[0]) 
        PET = str(row.iloc[1])
        CT_SCAN = str(row.iloc[2])
        RADIOLOGY_REPORT = str(row.iloc[3])
        TISSUE_SLIDE = str(row.iloc[4])
        BIOPSY_REPORT = str(row.iloc[5])

        if PET != "nan" and files != "":
            files +=  ", " + ID + "_" + PET
        elif PET != "nan" and files == "":
            files += ID + "_" + PET
        
        if CT_SCAN != "nan" and files != "":
            files += ", " + ID + "_" + CT_SCAN
        elif CT_SCAN != "nan" and files == "":
            files +=  ID + "_" + CT_SCAN

        if RADIOLOGY_REPORT != "nan" and files != "":
            files +=  ", " + ID + "_" + RADIOLOGY_REPORT
        elif RADIOLOGY_REPORT != "nan" and files == "":
            files += ID + "_" + RADIOLOGY_REPORT

        if TISSUE_SLIDE != "nan" and files != "":
            files += ", " + ID + "_" + TISSUE_SLIDE
        elif TISSUE_SLIDE != "nan" and files == "":
            files += ID + "_" + TISSUE_SLIDE

        if BIOPSY_REPORT != "nan" and files != "":
            files += ", " + ID + "_" + BIOPSY_REPORT
        elif BIOPSY_REPORT != "nan" and files == "":
            files += ID + "_" + BIOPSY_REPORT


        query = f"Hi, i have this files available: {files}, could you help me with the diagnosis?"
        query_name = f"query_{i}"
        dataset[query_name] = query

    data_tools =  pd.read_excel("gt_tools_diagnosis.xlsx")
    GT_tools = {}

    for i, (_, row) in enumerate(data_tools.iterrows(), start=1):
        toolsGT = row.iloc[0]
        tool = [t.strip() for t in ast.literal_eval(toolsGT)]
        query_name = f"query_{i}"
        GT_tools[query_name] = tool
    
    return GT_tools, dataset 

def levenshtein_distance(predicted, grounf_truth):
    
    """
    Calcola la distanza di edit tra due liste di stringhe.

    """
    size_x = len(predicted) + 1
    size_y = len(grounf_truth) + 1
    matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]

    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if predicted[x-1] == grounf_truth[y-1]:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1],    # Nessun costo se uguali
                    matrix[x][y-1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,  # Cancellazione
                    matrix[x-1][y-1] + 1, # Sostituzione
                    matrix[x][y-1] + 1   # Inserimento
                )

    levenshtein = matrix[size_x-1][size_y-1]
    return levenshtein

def correct_tool_pos(predicted, ground_truth):
    
    couples = int(sum(1 for a, b in zip(predicted, ground_truth) if a == b))
    if len(ground_truth) != 0:
        norm = couples / len(ground_truth)
    elif len(ground_truth) == 0 and len(predicted) == 0:
        norm = 1
    else:
        norm = 0
    return norm

def first_E(predicted, ground_truth): 
    
    first_error = -1
    min_len = min(len(predicted), len(ground_truth))
    for i in range(min_len):
        p = predicted[i]
        g = ground_truth[i]
        if p != g:
            first_error = i
            break
    if first_error == -1 and len(predicted) != len(ground_truth):
        first_error = min(len(predicted), len(ground_truth))
    first_error += 1  
    return first_error

def tool_match(predicted, ground_truth):
    
    match = 0
    if predicted == ground_truth:
        match = 1
    return match
def missed_tools(predicted, ground_truth):
    if not ground_truth:
        return 0
    
    gt_copy = list(ground_truth)
    for p in predicted:
        if p in gt_copy:
            gt_copy.remove(p) 
            
    return len(gt_copy) / len(ground_truth)

def invented_tools(predicted, ground_truth):
    gt_copy = list(ground_truth)
    invented = 0
    
    for p in predicted:
        if p in gt_copy:
            gt_copy.remove(p)
        else:
            invented += 1 
            
    return invented

def jason_maker(id, task, reasoning_effort, modello, obtained_files, web_search, query, output, tools, folder):
    

    record = {
        "ID": id,
        "task" : task,
        "model" : modello,
        "reasoning_effort" : reasoning_effort,
        "web_search" : web_search,
        "query": query,
        "tool_calls" : tools,
        "obtained_files" : obtained_files,
        "content": output['messages'][-1].content
        }
    
    output_file = os.path.join(folder, f"{task}_{modello}_output.jsonl")
    with open(output_file, 'a', encoding='utf-8') as f: #a serve per l'ordine
        f.write(json.dumps(record) + '\n')

    return record


def jason_metrics(task, modello, reasoning_effort, tool_acc, mean_tool_pos, mean_levenshtein, mean_missed_tools, mean_allucinated_tools, Tot_input_token, Tot_output_token, h, m, s, folder):
    


    final_metrics = {
        "task" : task,
        "model" : modello,
        "reasoning_effort" : reasoning_effort,
        "tool_accuracy": tool_acc,
        "tool_position_accuracy": mean_tool_pos,
        "avg_levenshtein_distance": mean_levenshtein,
        "avg_missed_tools": mean_missed_tools,
        "avg_hallucinated_tools": mean_allucinated_tools,
        "time" : f"{int(h)} hours {int(m)} minutes and {int(s)}seconds",
        "usage_and_costs": {
            "total_input_tokens": Tot_input_token,
            "total_output_tokens": Tot_output_token,
            "input_cost_usd": Tot_input_token * 0.00000175,
            "output_cost_usd": Tot_output_token * 0.000014,
            "total_cost_usd": (Tot_input_token * 0.00000175) + (Tot_output_token * 0.000014)
        }
    }
    metrics_file = os.path.join(folder, f"{task}_{modello}_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)
        print(f"\nMetrics successfully saved to {metrics_file}")

def save_failed_queries(failed_list, modello, folder, task_name="diagnosis"):
    if not failed_list:
        print("All queries were processed successfully.")
        return   

    output_filename = os.path.join(folder, f"failed_queries_{task_name}_{modello}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(failed_list, f, indent=4)
        
    print(f"{len(failed_list)} queries failed completely.")
    print(f"Saved failed queries to: {output_filename}.")



def token_usage(result):
    current_input = 0
    current_output = 0
    
    for message in result["messages"]:
        if isinstance(message, AIMessage) and getattr(message, 'usage_metadata', None):
            current_input += message.usage_metadata['input_tokens']
            current_output += message.usage_metadata['output_tokens']
    return current_input, current_output



if __name__ == "__main__":

    GT_tools, dataset = load_data()
    KEYS_LIST = load_keys()

    mt = 0  
    it = 0
    ctm = 0
    ld = 0
    ctp = 0
    
    j = 0
    v = 0
    c = 0
    K = 0
    Tot_input_token = 0
    Tot_output_token = 0
    
    d_model = create_model(KEYS_LIST[j])
    start_time = time.perf_counter()
    
    successful_runs = 0
    failed_queries = []
    
    stop_eval = False 
    last_query_index = 0

    for b, (query_name, query) in enumerate(dataset.items(), start=1):
        
        last_query_index = b 
        
        obtained_files.clear()
        inputs = {"messages": [("user", query)]}

        max_retries = 4
        invoke_successful = False
        
        for attempt in range(max_retries):
            try:
                result = diagnosis.invoke(inputs)
                invoke_successful = True
                c += 1
                if c < 2:
                    model_name = result['messages'][-1].response_metadata.get('model_name').replace(":", "_")
                    OUTPUT_FOLDER = f"{task}_{Prompt}_{model_name}_evaluation"
                    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                break  
            except Exception as e:
                error_msg = str(e).lower()
                print(f"Errore iterazione {b} tentativo: {attempt + 1}/{max_retries}): {e}") #parte da zero attempt
                
                if any(keyword in error_msg for keyword in ["429", "rate limit", "quota", "token", "usage"]):
                    print("Limite API raggiunto, chiave successiva")
                    j += 1
                    
                    if j >= len(KEYS_LIST):
                        print("Chiavi Esaurite, ricomincio")
                        j = 0 
                        v += 1
                        if v > 1:
                            stop_eval = True
                            break 

                    K += 1 
                    d_model = create_model(KEYS_LIST[j])
                    continue
                
                
                else:
                    print("Errore generico")
                    time.sleep(3) 
        
        if stop_eval:
            break

        if not invoke_successful:
            print(f"Skippo la query {b}")
            failed_queries.append({"query_name": query_name, "query": query})
            continue 
        

        successful_runs += 1 

        token_in, token_out = token_usage(result)
        Tot_input_token += token_in
        Tot_output_token += token_out
           
        tool_used = extract_tool_calls(result)
        jason_maker(b, task, reasoning_effort, model_name, obtained_files, web_search, query, result, tool_used, OUTPUT_FOLDER)

        ctm += tool_match(tool_used, GT_tools[query_name])
        ctp += correct_tool_pos(tool_used, GT_tools[query_name])
        ld += levenshtein_distance(tool_used, GT_tools[query_name])
        mt += missed_tools(tool_used, GT_tools[query_name])
        it += invented_tools(tool_used, GT_tools[query_name])  

        print(f"-----{model_name} Iterazione {b}: api key n° {j+1}")

    
    end_time = time.perf_counter()
    total_time_seconds = end_time - start_time
    minutes, seconds = divmod(total_time_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if successful_runs > 0:
        tool_acc = ctm / successful_runs
        mean_tool_pos = ctp / successful_runs 
        mean_levenshtein = ld / successful_runs
        mean_missed_tools = mt / successful_runs
        mean_allucinated_tools = it / successful_runs
        
        jason_metrics(task, model_name, reasoning_effort,  tool_acc, mean_tool_pos, mean_levenshtein, mean_missed_tools, mean_allucinated_tools, Tot_input_token, Tot_output_token, hours, minutes, seconds, OUTPUT_FOLDER)
    


    save_failed_queries(failed_queries, model_name, OUTPUT_FOLDER, task_name="diagnosis")


    raw_state = {
        "index": last_query_index,
        "stop": stop_eval,
        "successful_runs": successful_runs,
        "model": model_name,
        "ctm": ctm,
        "ctp": ctp,
        "ld": ld,
        "mt": mt,
        "it": it,
        "Tot_input_token": Tot_input_token,
        "Tot_output_token": Tot_output_token,
    }
    
    raw_state_file = os.path.join(OUTPUT_FOLDER, f"{task}_metrics_raw_{model_name}.json")
    with open(raw_state_file, "w", encoding="utf-8") as f:
        json.dump(raw_state, f, indent=4)
        print("\n metriche grezze ottenute")

    
    if stop_eval:
        print(f"codice interrotto forzatamente")