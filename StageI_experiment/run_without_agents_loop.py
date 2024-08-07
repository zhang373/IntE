import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_file_path)
os.chdir(parent_directory)
print("There is an os process in the head of the file and we have move the working path to: ", os.getcwd())

from StageI_experiment.run_with_agents_loop import save_data_to_directory
from DataSet.dataset_processing import Get_Process_Full_Data
from stage_I import build_R2_prompt_Single_Question_without_agents
import re




Dataset_Names = ['Computer_Human_Interaction','Personal_Financial_Literacy']
for Dataset_Name in Dataset_Names:
    Q = Get_Process_Full_Data(Dataset_Name=Dataset_Name, Print_Flag=False)
    print(f"Current Dataset is {Dataset_Name} and we are doing the reviewing")


    Draft_history_dict = {}
    Outcome_history = {}


    for item in Q.keys():
        print(f"\nWe are Processing: {item} in Dataset {Dataset_Name}!")
        conditions = re.findall(r'\((.*?)\)', Q[item][[item for item in list(Q[item].keys()) if item.startswith("Question_")][0]])
        Draft_history_dict[item], Outcome_history[item] = None, None #build_R2_prompt_Single_Question_without_agents(Q[item], conditions, Dataset_Name, Print_Flag=False)
    print("\nWe have finish all the process in stage I, Go ahead to access all the data!")

    StageI_indenpendt_experiment_exp_data = {"Draft_history_dict": Draft_history_dict, "Outcome_history": Outcome_history}
    save_data_to_directory(data=StageI_indenpendt_experiment_exp_data, directory='StageI_experiment/result/',
                           file_name="Human_"+Dataset_Name, Clear=True)