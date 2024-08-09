import json
import pandas as pd
import os

class class_division:
    def __init__(self):
        return
    # TODO: divide user into different groups
    def categorize_users(self, high_skill=None, medium_skill=None, low_skill=None):
        high_skill = ["zws_1", "zws_2"] if high_skill==None else high_skill
        medium_skill = [] if medium_skill==None else medium_skill
        low_skill = [] if low_skill==None else low_skill
        categorized_users = {
            "High": high_skill,
            "Medium": medium_skill,
            "Low": low_skill
        }
        return categorized_users

    # TODO:divide questions into different subtasks
    def extract_tasks_and_subtasks(self, Dataset_Names=None):
        Task_index_dict = {}
        if Dataset_Names==None:
            Dataset_Names = ['Computer_Human_Interaction', 'Personal_Financial_Literacy']
        for Dataset_Name in Dataset_Names:
            if Dataset_Name == 'Computer_Human_Interaction':
                # Attention: you can modify this to change subtask
                SubTask_1 = {"name": "System_Engineering", "index": ["Question_1", "Question_2", "Question_4"]}
                SubTask_2 = {"name": "Human_Computer_Technology", "index": ["Question_3", "Question_5"]}
                Task_Discovery = {
                    "SubTask_1": SubTask_1,
                    "SubTask_2": SubTask_2,
                }
                Task_index_dict["Task_Discovery"] = {"name": Dataset_Name, "index":Task_Discovery}
            elif Dataset_Name == 'Personal_Financial_Literacy':
                # Attention: you can modify this to change subtask
                SubTask_1 = {"name": "x1", "index": ["Question_1", "Question_2", "Question_3"]}
                SubTask_2 = {"name": "x2", "index": ["Question_4", "Question_5"]}
                SubTask_3 = {"name": "x2", "index": ["Question_6", "Question_7"]}
                Task_Summary = {
                    "SubTask_1": SubTask_1,
                    "SubTask_2": SubTask_2,
                    "SubTask_3": SubTask_3,
                }
                Task_index_dict["Task_Summary"] = {"name": Dataset_Name, "index": Task_Summary}
            else:
                raise ValueError(f"You have input a wrong Dataset_Name: {Dataset_Name}")
        return Task_index_dict


    def build_user_subtask_mapping(self,):
        Task_index_dict = self.extract_tasks_and_subtasks()
        categorized_users = self.categorize_users()

        json_dict = {}

        Dataset_Names = [task["name"] for task in Task_index_dict.values()]
        subtask_indexs = [task["index"] for task in Task_index_dict.values()]

        for (Dataset_Name, subtask_index) in zip(Dataset_Names, subtask_indexs):
            for user_type in categorized_users.keys():
                for subtask in subtask_index.keys():
                    sub_task_name = subtask_index[subtask]['name']
                    sub_task_index = subtask_index[subtask]['index']
                    final_dict_key = str((user_type, sub_task_name))
                    data_buffer = []
                    user_names = categorized_users[user_type]

                    for user_name in user_names:
                        data_unit = {}
                        for exp_type in ['Human', 'Agent']:
                            filename = "result/"+"StageI_independent_experiment_data_" + exp_type + \
                                       Dataset_Name + "_" + user_name
                            with open(filename, 'r', encoding='utf-8') as file:
                                # 将文件内容加载成字典
                                data_dict = json.load(file)
                            extracted_data = {key: {inner_key: data_dict[key][inner_key] for inner_key in sub_task_index}
                                              for key in data_dict}
                            data_unit[exp_type] = extracted_data
                        data_buffer.append(data_unit)

                    json_dict[final_dict_key] = data_buffer

        return json_dict

    def eval_json_dict(self):
        json_dict = self.build_user_subtask_mapping()
        score_dict = {}
        for key in json_dict.keys():
            temp_dict = {}
            for exp_type in ['Human', 'Agent']:
                dataunit_list = json_dict[key]
                tryed_times = 0
                success_duration = 0
                person_amount = len(dataunit_list)

                for dataunit in dataunit_list:
                    tryed_times_per_person = 0
                    success_duration_per_person = 0
                    for item in dataunit[exp_type]["All_Drafts_dict"].values():
                        tryed_times_per_person += len(item)
                    for item in dataunit[exp_type]["Draft_history_dict"].values():
                        success_duration_per_person += len(item)
                    total_question_num = len(dataunit[exp_type]["All_Drafts_dict"].keys())
                    tryed_times_per_person = tryed_times_per_person/total_question_num
                    success_duration_per_person = success_duration_per_person/total_question_num
                    tryed_times += tryed_times_per_person
                    success_duration += success_duration_per_person

                tryed_times = tryed_times/person_amount
                success_duration = success_duration/person_amount
                temp_dict[exp_type] = (tryed_times, success_duration)
            score_dict[key] = temp_dict

        return score_dict

    def dict_to_tables(self, final_dict, output_dir='result'):
        # 创建输出目录如果它不存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 初始化两个空的DataFrame
        human_df = pd.DataFrame()
        agent_df = pd.DataFrame()

        # 遍历final_dict
        for key, values in final_dict.items():
            # 解析键
            user_type, sub_task_name = eval(key.strip('()'))

            # 更新human_df
            if 'Human' in values:
                human_df.at[user_type, sub_task_name] = values['Human']

            # 更新agent_df
            if 'Agent' in values:
                agent_df.at[user_type, sub_task_name] = values['Agent']

        # 重置列名索引
        human_df.columns.name = 'Sub-task'
        agent_df.columns.name = 'Sub-task'

        # 保存到文件
        human_df.to_csv(os.path.join(output_dir, 'human_table.csv'))
        agent_df.to_csv(os.path.join(output_dir, 'agent_table.csv'))

        return human_df, agent_df

    def process_json_data_files(self):
        score_dict = self.eval_json_dict()
        self.dict_to_tables(final_dict=score_dict)