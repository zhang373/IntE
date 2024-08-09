import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_file_path)
os.chdir(parent_directory)
print("There is an os process in the head of the file and we have move the working path to: ", os.getcwd())

from DataSet.dataset_processing import Get_Process_Full_Data
from stage_I import build_R2_prompt_Single_Question
import re
import shutil
import json
import copy

# Some useful function
def save_data_to_directory(data, directory="result", file_name="test_name", Clear=False):
    file_name = "StageI_independent_experiment_data_" + file_name + ".json"
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    if Clear:
        # Clear the contents of the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                return f"Error: Failed to delete {file_path}. Reason: {e}"

    # Save the data to a JSON file
    try:
        with open(os.path.join(directory, file_name), 'w') as f:
            json.dump(data, f)
        return f"Data has been saved to {os.path.join(directory, file_name)}"
    except Exception as e:
        return f"Error: Failed to save data. Reason: {e}"

if __name__ == "__main__":
    Dataset_Names = ['Computer_Human_Interaction','Personal_Financial_Literacy']
    file_name = copy.deepcopy(input("Please input your name here:"))

    for Dataset_Name in Dataset_Names:
        Q = Get_Process_Full_Data(Dataset_Name=Dataset_Name, Print_Flag=False)
        print(f"Current Dataset is {Dataset_Name} and we are doing the reviewing")

        prompt_R2_dict = {}
        Predefined_Task_Information_Setting_dict = {}
        Draft_history_dict = {}
        Review_history_dict = {}
        Final_Evaluation_Result_dict = {}
        All_Drafts_dict = {}
        All_Outcomes_dict = {}
        for item in Q.keys():
            print(f"\nWe are Processing: {item} in Dataset {Dataset_Name}!")
            conditions = re.findall(r'\((.*?)\)', Q[item][[item for item in list(Q[item].keys()) if item.startswith("Question_")][0]])
            print("We are processing the question: ", Q[item][[item for item in list(Q[item].keys()) if item.startswith("Question_")][0]])
            print(f"And current conditions are {conditions}")
            prompt_R2_dict[item], Predefined_Task_Information_Setting_dict[item], \
            Draft_history_dict[item], Review_history_dict[item], Final_Evaluation_Result_dict[item], All_Drafts_dict[item], All_Outcomes_dict[item]= \
                build_R2_prompt_Single_Question(Q[item], conditions, Dataset_Name, Print_Flag=False, Test_Mode=True)
        print("\nWe have finish all the process in stage I, Go ahead to access all the data!")

        StageI_indenpendt_experiment_exp_data = {"prompt_R2_dict":prompt_R2_dict, "Predefined_Task_Information_Setting_dict": Predefined_Task_Information_Setting_dict,
                                                 "Draft_history_dict": Draft_history_dict, "Review_history": Review_history_dict, "Final_Evaluation_Result": Final_Evaluation_Result_dict,
                                                 "All_Outcomes":All_Outcomes_dict, "All_Drafts":All_Drafts_dict}
        save_data_to_directory(data=StageI_indenpendt_experiment_exp_data, directory='StageI_experiment/result/',
                               file_name="Agent_"+Dataset_Name+"_"+str(file_name), Clear=False)