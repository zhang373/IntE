from Robort1 import call_with_messages_Qwen, Draft_Modifier, build_R1_init_prompt
from Robort2 import Cal_sim
from DataSet.dataset_processing import Get_Process_Full_Data
import re



def build_R2_prompt_Single_Question(Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy', Print_Flag=True):
    if Print_Flag:
        pass
        print("Current Q_SQ.keys() ", Q_SQ.keys())
        print("Current conditions: ", conditions)
    Predefined_Task_Information_Setting_Prompt, Predefined_Task_Information_Setting = build_R1_init_prompt(Dataset_Name, conditions)
    Draft_history = []
    Review_history = []
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

        user_input = input("User input reviewing, enter ['exit', 'quit'] to finish: ")
        if user_input.lower() in ['exit', 'quit']:  # Check if the user wants to quit
            break
        Review_history.append(user_input)
        agent_description = "You are a helpful assistant and know will about how to modify past data based on reviewing information."
        current_draft = Draft_Modifier(model_name="qwen-turbo", Draft=Draft_history,
                                       ReviewInfo=Review_history, iter_loop=1,
                                       Predefined_Task_Information_Setting=Predefined_Task_Information_Setting)
        # print("Current Draft Prompt after modified by LLM Agent: ", current_draft)
        Draft_history.append(current_draft)

    result = Draft_history[-1]
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

if __name__ == "__main__":
    Dataset_Name = 'Personal_Financial_Literacy'
    Q = Get_Process_Full_Data(Dataset_Name=Dataset_Name, Print_Flag=False)
    prompt_R2_dict, Predefined_Task_Information_Setting_dict = build_R2_prompt_Question_Set(Dataset_Name, Q)
    print(prompt_R2_dict)