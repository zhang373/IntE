from Robort1 import call_with_messages_Qwen, Draft_Modifier, build_R1_init_prompt
from Robort2 import Cal_sim
from DataSet.dataset_processing import Get_Process_Full_Data
import re
import tkinter as tk
from tkinter import messagebox


def build_R2_prompt_Single_Question(Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy', Print_Flag=True, Test_Mode=False):
    if Print_Flag:
        pass
        print("Current Q_SQ.keys() ", Q_SQ.keys())
        print("Current conditions: ", conditions)
    Predefined_Task_Information_Setting_Prompt, Predefined_Task_Information_Setting = build_R1_init_prompt(Dataset_Name, conditions)

    All_Drafts = []  # New list to store all drafts
    All_Outcomes = []  # New list to store all outcomes

    Draft_history = []
    Review_history = []
    Outcome_history = []
    R2_result_Draft = call_with_messages_Qwen(content=Predefined_Task_Information_Setting_Prompt,
                                              agent_des=f"You are a useful agent and works will in {Dataset_Name} domain.",
                                              model_name="qwen-turbo")["output"]["choices"][0]["message"]["content"]
    Draft_history.append(R2_result_Draft)
    if Print_Flag:
        pass
        print("Current initial Draft_history is: \n", Draft_history)

    while True:
        # Use R2 to return demo result
        out = Cal_sim(Q_SQ[[item for item in list(Q_SQ.keys()) if item.endswith("_Prompt")][0]][0][0],  # the pair
                      Q_SQ[[item for item in list(Q_SQ.keys()) if item.startswith("Question_")][0]],    # the question
                      Cal_mode_Singel=False, CoT=Draft_history[-1],
                      Predefined_Task_Information_Setting=Predefined_Task_Information_Setting,
                      StageI=True).output.choices[0].message.content
        print("The outcome by using current built prompt\n", out)
        Outcome_history.append(out)

        # TODO: modify the process here! and deal with the bugs!
        print("\n Current Draft Prompt after modified by LLM Agent: \n", Draft_history[-1])

        user_input = input("User input reviewing, enter ['exit', 'restart'] to finish or restart: ")
        if user_input.lower() == 'restart':
            All_Drafts.append(Draft_history)
            All_Outcomes.append(Outcome_history)
            Draft_history.clear()
            Outcome_history.clear()
            Draft_history.append(R2_result_Draft)

            continue  # Restart the loop
        elif user_input.lower() in ['exit', 'quit']:  # Check if the user wants to quit
            break
        Review_history.append(user_input)
        agent_description = "You are a helpful assistant and know will about how to modify past data based on reviewing information."
        current_draft = Draft_Modifier(model_name="qwen-turbo", Draft=Draft_history,
                                       ReviewInfo=Review_history, iter_loop=1,
                                       Predefined_Task_Information_Setting=Predefined_Task_Information_Setting)

        Draft_history.append(current_draft)

    result = Draft_history[-1]

    if Test_Mode:
        return result, Predefined_Task_Information_Setting, Draft_history, Review_history, Outcome_history, All_Drafts, All_Outcomes
    else:
        return result, Predefined_Task_Information_Setting

def build_R2_prompt_Question_Set(Dataset_Name, Q):
    prompt_R2_dict = {}
    Predefined_Task_Information_Setting_dict = {}
    for item in Q.keys():
        print(f"We are Processing: {item} in Dataset {Dataset_Name}!")
        conditions = re.findall(r'\((.*?)\)',
                                Q[item][[item for item in list(Q[item].keys()) if item.startswith("Question_")][0]])
        prompt_R2_dict[item], Predefined_Task_Information_Setting_dict[item] = \
            build_R2_prompt_Single_Question(Q[item], conditions, Dataset_Name, Print_Flag=False)
    print("We have finish all the process in stage I, Go ahead to access all the data!")
    # print(prompt_R2_list)
    return prompt_R2_dict, Predefined_Task_Information_Setting_dict

def build_draft_prompt_without_agents(draft_prompts, sample, conditions, add_structure=False):
    draft_prompts_out = draft_prompts
    draft_prompts_out += f"give the c-sts of current test pairs: {str(sample)} under the conditions:{conditions}. " \
                         f"You should response in the following python "
    if add_structure:
        draft_prompts_out+="dict format:{'Reason': your reason, 'Score': {'condition1': score1, 'condition2': score2, ...} " \
                      "You only need to give me the dict and must follow the format!"
    return draft_prompts_out

def build_R2_prompt_Single_Question_without_agents(Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy', Print_Flag=True):
    if Print_Flag:
        pass
        print("Current Q_SQ.keys() ", Q_SQ.keys())
        print("Current conditions: ", conditions)

    All_Drafts = []  # New list to store all drafts
    All_Outcomes = []  # New list to store all outcomes

    Draft_history = []
    Outcome_history = []
    print("In this human-computer interaction comparative study, your task is to craft a set of prompts "
          "that will enable the LLM (Large Language Model) to acquire the capability of evaluating text similarity under various conditions. "
          "Specifically, the LLM should be able to assess the semantic similarity between pairs of texts based on conditional information. "
          "It is crucial that the LLM develops a general ability that performs well across different tasks (universal problem-solving). "
          "Please design a comprehensive general prompt to achieve this objective. "
          "We have add the condition and sample into the prompt, you just need to finish the rest of it. Please do not add test sample!")
    print("Current condition we are working on: \n", conditions)
    sample = Q_SQ[[item for item in list(Q_SQ.keys()) if item.endswith("_Prompt")][0]][0][0]
    print("You can test your prompt on this demo data pair:\n", sample)

    if Print_Flag:
        pass
        print("Current initial Draft_history is: \n", Draft_history)

    while True:
        draft_prompt = input("Please input your current prompt: ")
        Draft_history.append(draft_prompt)
        draft_prompt = build_draft_prompt_without_agents(draft_prompts=draft_prompt,sample=sample,conditions=conditions)
        R2_result = call_with_messages_Qwen(content=draft_prompt,
                                                  agent_des=f"You are a useful agent and works will in {Dataset_Name} domain.",
                                                  model_name="qwen-turbo")["output"]["choices"][0]["message"]["content"]
        print("Current prompt will lead to this outcome: ", R2_result)
        Outcome_history.append(R2_result)

        user_input = input("\n User input reviewing, enter ['exit', 'restart'] to finish or restart, else press any key and enter ")
        if user_input.lower() == 'restart':
            All_Drafts.append(Draft_history)
            All_Outcomes.append(Outcome_history)
            Draft_history.clear()
            Outcome_history.clear()
            continue  # Restart the loop
        elif user_input.lower() in ['exit', 'quit']:  # Check if the user wants to quit
            break

    return Draft_history, Outcome_history, All_Drafts, All_Outcomes


if __name__ == "__main__":
    # 主窗口初始化
    window = tk.Tk()
    window.title("Prompt Writing Application")
    Dataset_Names = ['Personal_Financial_Literacy']
    for Dataset_Name in Dataset_Names:
        Q = Get_Process_Full_Data(Dataset_Name=Dataset_Name, Print_Flag=False)
        print(f"Current Dataset is {Dataset_Name} and we are doing the reviewing")

        Draft_history_dict = {}
        Outcome_history_dict = {}
        All_Drafts_dict = {}
        All_Outcomes_dict = {}

        for item in Q.keys():
            print(f"\nWe are Processing: {item} in Dataset {Dataset_Name}!")
            conditions = re.findall(r'\((.*?)\)', Q[item][[item for item in list(Q[item].keys()) if item.startswith("Question_")][0]])
            # setup_page2(window, Q[item], conditions, Dataset_Name='Personal_Financial_Literacy')
            break