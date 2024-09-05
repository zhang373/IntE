import tkinter as tk
import os
import json
file_path = "StageI_experiment/result/username_entry.txt"

current_file_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_file_path)
os.chdir(parent_directory)
print("There is an os process in the head of the file and we have move the working path to: ", os.getcwd())
from stage_I import build_draft_prompt_without_agents
from Robort1 import call_with_messages_Qwen, Draft_Modifier, build_R1_init_prompt
from Robort2 import Cal_sim
class Page3App:
    def __init__(self, master, Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy'):
        self.master = master
        self.master.title("Page 3 - Prompt Writing with adversarial loop")

        # 用户信息区域
        with open(file_path, 'r', encoding='utf-8') as file:
            self.user_register_name = file.read()

        # 定义一些全局变量
        self.All_Drafts = []        # New list to store all drafts
        self.All_Outcomes = []      # New list to store all outcomes
        self.All_reviews = []       # New list to store all reviews

        self.Draft_history = []
        self.Review_history = []
        self.Outcome_history = []

        # 宏观信息区域
        self.macro_info_label = tk.Label(self.master, text="The general instructions are here!", anchor="nw",
                                         justify="left")
        self.macro_info_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5)
        self.macro_info_UI = tk.Text(self.master, height=5, width=50)
        self.macro_info_UI.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

        # 历史prompt和output区域
        self.prompt_history_label = tk.Label(self.master, text="The history prompt are here!", anchor="nw",
                                             justify="left")
        self.prompt_history_label.grid(row=2, column=0, padx=10, pady=(5, 0))
        self.prompt_history_UI = tk.Text(self.master, height=5, width=25)
        self.prompt_history_UI.grid(row=3, column=0, padx=10, pady=5)

        self.outcome_history_label = tk.Label(self.master, text="The history outcome are here!", anchor="nw",
                                              justify="left")
        self.outcome_history_label.grid(row=2, column=1, padx=10, pady=(5, 0))
        self.outcome_history_UI = tk.Text(self.master, height=5, width=25)
        self.outcome_history_UI.grid(row=3, column=1, padx=10, pady=5)


        # 当前prompt和output区域
        self.current_prompt_label = tk.Label(self.master, text="The current input prompt are here!", anchor="nw",
                                             justify="left")
        self.current_prompt_label.grid(row=4, column=0, padx=10, pady=(5, 0))
        self.current_outcome_label = tk.Label(self.master, text="The current outcome are here!", anchor="nw",
                                              justify="left")
        self.current_outcome_label.grid(row=4, column=1, padx=10, pady=(5, 0))

        self.current_outcome_UI = tk.Text(self.master, height=5, width=25)
        self.current_prompt_UI = tk.Text(self.master, height=5, width=25)
        self.current_prompt_UI.grid(row=5, column=0, padx=10, pady=5)
        self.current_outcome_UI.grid(row=5, column=1, padx=10, pady=5)

        # 输入框
        self.prompt_input_label = tk.Label(self.master,
                                           text="Please input your review here! \n enter ['exit', 'restart'] to finish or restart.",
                                           anchor="nw", justify="left")
        self.prompt_input_label.grid(row=6, column=0, columnspan=2, padx=10, pady=(5, 0))
        self.prompt_input_UI = tk.Entry(self.master, width=50)
        self.prompt_input_UI.grid(row=7, column=0, columnspan=2, padx=10, pady=5)

        # 修改结果展示框
        self.processed_prompt_label = tk.Label(self.master, text="Renewed input prompt is shown here!", anchor="nw",
                                               justify="left")
        self.processed_prompt_label.grid(row=8, column=0, columnspan=2, padx=10, pady=(5, 0))
        self.processed_prompt_input_UI = tk.Text(self.master, height=5, width=25)
        self.processed_prompt_input_UI.grid(row=9, column=0, columnspan=2, padx=10, pady=5)

        # 输入状态显示框
        self.input_status_label = tk.Label(self.master, text="Please input review and click submit", anchor="nw",
                                           justify="left")
        self.input_status_label.grid(row=10, column=1, padx=10, pady=(5, 0))
        self.input_statu_UI = tk.Text(self.master, height=5, width=25)
        self.input_statu_UI.grid(row=11, column=1, padx=10, pady=5)

        # 提交按钮
        self.submit_button_UI = tk.Button(self.master, text="Submit Review", command=self.handle_input)
        self.submit_button_UI.grid(row=11, column=0, padx=10, pady=5)

        # 初始化界面
        self.Dataset_Name = 'Personal_Financial_Literacy'
        self.conditions = conditions
        self.Q_SQ = Q_SQ
        self.sample = self.Q_SQ[[item for item in list(Q_SQ.keys()) if item.endswith("_Prompt")][0]][0][0]

        self.init_blocks()
        self.init_initial_information_block()

    def init_initial_information_block(self):
        self.input_statu_UI.insert(tk.END, "Please input prompt and click submit\n")
        self.macro_info_UI.insert(tk.END, "The general instructions are here!\n")
        macro_info_base = "In this human-computer interaction comparative study, your task is to craft a set of prompts " \
                          "that will enable the LLM (Large Language Model) to acquire the capability of evaluating text similarity under various conditions. " \
                          "Specifically, the LLM should be able to assess the semantic similarity between pairs of texts based on conditional information. " \
                          "It is crucial that the LLM develops a general ability that performs well across different tasks (universal problem-solving). " \
                          "Please design a comprehensive general prompt to achieve this objective with the help of our multi agent adversarial loop. " \
                          "We have added the condition and sample into the prompt, you just need to finish the rest of it. "
        macro_info_base += ("\nCurrent condition we are working on are: \n" + str(self.conditions))
        macro_info_base += ("\nYou can test your prompt on this demo data pair:\n" + str(self.sample))
        self.macro_info_UI.insert(tk.END, f"{macro_info_base}\n")

        Predefined_Task_Information_Setting_Prompt, Predefined_Task_Information_Setting = build_R1_init_prompt(
            self.Dataset_Name, self.conditions)
        R2_result_Draft = call_with_messages_Qwen(content=Predefined_Task_Information_Setting_Prompt,
                                                  agent_des=f"You are a useful agent and works will in {self.Dataset_Name} domain.",
                                                  model_name="qwen-turbo")["output"]["choices"][0]["message"]["content"]
        self.prompt_history_UI.insert(tk.END, f"{self.Draft_history}")
        self.current_prompt_UI.insert(tk.END, R2_result_Draft)
        self.Draft_history.append(R2_result_Draft)

        # Use R2 to return demo result
        out = Cal_sim(self.Q_SQ[[item for item in list(self.Q_SQ.keys()) if item.endswith("_Prompt")][0]][0][0],  # the pair
                      self.Q_SQ[[item for item in list(self.Q_SQ.keys()) if item.startswith("Question_")][0]],    # the question
                      Cal_mode_Singel=False, CoT=self.Draft_history[-1],
                      Predefined_Task_Information_Setting=Predefined_Task_Information_Setting,
                      StageI=True).output.choices[0].message.content
        self.outcome_history_UI.insert(tk.END, f"{self.Outcome_history}")
        self.current_outcome_UI.insert(tk.END, out)
        self.Outcome_history.append(out)



    # 清空输出函数
    def clear_outputs(self):
        # self.macro_info_UI.delete('1.0', tk.END)
        self.prompt_history_UI.delete('1.0', tk.END)
        self.outcome_history_UI.delete('1.0', tk.END)
        self.current_prompt_UI.delete('1.0', tk.END)
        self.current_outcome_UI.delete('1.0', tk.END)
        self.input_statu_UI.delete('1.0', tk.END)
        self.processed_prompt_input_UI.delete('1.0', tk.END)

    def init_blocks(self):
        self.processed_prompt_label.config(text="Processed input prompt is shown here!")
        self.current_prompt_label.config(text="The current input prompt are here!")
        self.current_outcome_label.config(text="The current outcome are here!")
        self.prompt_history_label.config(text="The history prompt are here!")
        self.outcome_history_label.config(text="The history outcome are here!")
        self.prompt_input_label.config(text="Please input your prompt here! \n enter ['exit', 'restart'] to finish or restart.")
        self.input_status_label.config(text="Please input prompt and click submit")

    # 处理输入
    def handle_input(self):
        user_input = self.prompt_input_UI.get()
        if user_input.lower() == 'restart':
            """
            All_Drafts.append(Draft_history)
            All_Outcomes.append(Outcome_history)
            Draft_history.clear()
            Outcome_history.clear()
            
            continue  # Restart the loop
            """
            self.All_Drafts.append(self.Draft_history)
            self.All_Outcomes.append(self.Outcome_history)
            self.All_reviews.append(self.Review_history)
            self.Draft_history.clear()
            self.Outcome_history.clear()
            self.Review_history.clear()

            self.clear_outputs()
            self.prompt_input_UI.delete(0, tk.END)
            self.init_initial_information_block()
            self.input_statu_UI.insert(tk.END, "We just Restarted! Reenter prompt and click!")

        elif user_input.lower() in ['exit', 'quit']:  # Check if the user wants to quit
            self.master.destroy()
            data_path = "StageI_experiment/result/" + self.user_register_name
            with open(os.path.join(data_path, "ALL_Draft_agent_prompt_data.json"), 'w') as f1:
                json.dump(self.All_Drafts, f1)
            with open(os.path.join(data_path, "ALL_Outcome_agent_prompt_data.json"), 'w') as f2:
                json.dump(self.All_Outcomes, f2)
            with open(os.path.join(data_path, "ALL_Review_agent_prompt_data.json"), 'w') as f3:
                json.dump(self.All_reviews, f3)
        else:
            self.input_statu_UI.insert(tk.END, "Prompt is being Processing!")
            draft_prompt = user_input
            self.Draft_history.append(draft_prompt)
            self.current_prompt_UI.insert(tk.END, draft_prompt)

            # 构建提示
            processed_prompt = build_draft_prompt_without_agents(draft_prompt, self.sample, self.conditions)
            self.processed_prompt_input_UI.insert(tk.END, processed_prompt)

            # 调用模型
            R2_result = call_with_messages_Qwen(content=processed_prompt,
                                                agent_des=f"You are a useful agent and works will in {self.Dataset_Name} domain.",
                                                model_name="qwen-turbo")["output"]["choices"][0]["message"]["content"]
            self.Outcome_history.append(R2_result)
            self.current_outcome_UI.insert(tk.END, R2_result)

            # 清除输入框内容
            # self.init_blocks()  # resubmit information
            self.outcome_history_UI.insert(tk.END, f"\n{self.Outcome_history}")
            self.prompt_history_UI.insert(tk.END, f"\n{self.Draft_history}")
            self.clear_outputs()  # clear all
            self.prompt_input_UI.delete(0, tk.END)
            self.input_statu_UI.delete(0, tk.END)
            self.input_statu_UI.insert(tk.END, "Please input prompt and click submit!")

def build_R2_prompt_Single_Question(Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy', Print_Flag=True, Test_Mode=False):

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

# 使用示例
def setup_page3_app(the_window, Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy'):
    Page3App(the_window, Q_SQ, conditions, Dataset_Name)

def setup_page3(the_windows, Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy'):
    page3_window = tk.Toplevel(the_windows)
    page3_window.title("Page 3 - Prompt Writing with adversarial loop")
    setup_page3_app(page3_window, Q_SQ, conditions, Dataset_Name)