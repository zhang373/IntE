import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import webbrowser
import os
import re
import shutil
from datetime import datetime

# 导入 setup_page2 函数
from UI_normal_prompt import setup_page2
from UI_Agent_Loop import setup_page3
# 获取并更改工作目录
current_file_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_file_path)
os.chdir(parent_directory)
print("There is an os process in the head of the file and we have moved the working path to: ", os.getcwd())
from DataSet.dataset_processing import Get_Process_Full_Data


# 用于跳转网页的函数
def open_link_IntE_Question():
    webbrowser.open("https://wj.qq.com/s2/14975363/pvp7/")

def open_link_UI_Guidance():
    webbrowser.open("https://docs.google.com/presentation/d/1Eu20DgWUiHdQgkGBKmrXZaKzECP4D3IWwEdpzsLGVcM/edit#slide=id.g2893fd1c069_2_0")

def open_link_Normal_Question():
    webbrowser.open("https://wj.qq.com/s2/15314813/54ae/")

def open_link_Agent_Guidance():
    webbrowser.open("https://wj.qq.com/s2/15314972/9mx1/")

# 用于处理选项点击的函数
def on_select(value, the_windows):
    cur_question_index = int(user_cur_question_entry.get()) - 1
    Q_SQ, conditions = exp_Q_SQ_list[cur_question_index], exp_condition_list[cur_question_index]
    if value == "1":
        show_page(2, the_windows, Q_SQ, conditions)
    elif value == "2":
        show_page(3, the_windows, Q_SQ, conditions)

# 用于显示特定页面的函数
def show_page(page_number, the_windows, Q_SQ, conditions):
    user_cur_question_entry.delete(0, tk.END)
    if page_number == 2:
        # 创建一个 Toplevel 窗口
        page2_window = tk.Toplevel(the_windows)
        page2_window.title("Page 2 - Prompt Writing Application")
        setup_page2(page2_window, Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy')  # 在新窗口中设置 Page 2
    elif page_number == 3:
        # 创建一个 Toplevel 窗口
        page3_window = tk.Toplevel(the_windows)
        page3_window.title("Page 2 - Prompt Writing Application")
        setup_page3(page3_window, Q_SQ, conditions, Dataset_Name='Personal_Financial_Literacy')  # 在新窗口中设置 Page 3
    else:
        setup_page1()

def register_user():
    if username_entry.get() == '':
        return

    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        with open(file_path, 'w') as file:
            # 可以在这里写入一些文本
            file.write(username_entry.get())
        if os.path.exists("StageI_experiment/result/"+username_entry.get()+"/") and os.path.isdir("StageI_experiment/result/"+username_entry.get()+"/"):
            # 删除目录及其内容
            shutil.rmtree("StageI_experiment/result/"+username_entry.get()+"/")
        os.makedirs("StageI_experiment/result/"+username_entry.get()+"/")

        timestamp_path = "StageI_experiment/result/" + username_entry.get() + '/time_record.txt'
        with open(timestamp_path, 'w', encoding='utf-8') as file:
            file.write(f"register_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        username_entry_statu.insert(tk.END, "You have registered!")

def register_question():
    if user_cur_question_entry.get() == '':
        return

    with open("StageI_experiment/result/username_entry.txt", 'r', encoding='utf-8') as file:
        user_register_name = file.read()
    timestamp_path = "StageI_experiment/result/" + user_register_name + '/time_record.txt'

    with open(timestamp_path, 'a', encoding='utf-8') as file:
        file.write(f"We currently working on the question: {user_cur_question_entry.get()}. Question 5 is the example case. \n")
        file.write(f"Question_Selection_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    Question_entry_statu.insert(tk.END, f"You have select question {user_cur_question_entry.get()}!")

# 设置Page1
def setup_page1():
    global username_entry, choice_var, username_entry_statu, user_cur_question_entry, Question_entry_statu

    # 用户名输入
    username_label = tk.Label(window, text="Enter your username, example case please enter 'User':")
    username_entry = tk.Entry(window, width=20)
    register_button = tk.Button(window, text="Register user entry", command=register_user)

    # 链接按钮
    link_button = tk.Button(window, text="Visit IntE Questionnaire Website", command=open_link_IntE_Question)
    guidance_button = tk.Button(window, text="Visit Guidance for Using this System", command=open_link_UI_Guidance)
    normal_button = tk.Button(window, text="Visit Normal System Questionnaire Website", command=open_link_Normal_Question)
    agent_button = tk.Button(window, text="Visit Agent System Questionnaire Website", command=open_link_Agent_Guidance)

    # 问题选择框
    user_cur_question_label = tk.Label(window, text="Enter your assigned question (choice in only 1,2,3,4), Example case is 5: ")
    user_cur_question_entry = tk.Entry(window, width=20)
    question_sure_button = tk.Button(window, text="Register Question entry", command=register_question)

    # 选项
    choice_var = tk.StringVar(value="1")
    option1 = tk.Radiobutton(window, text="Prompt Writing with traditional GPT", variable=choice_var, value="1",
                             command=lambda: on_select(choice_var.get(), window))
    option2 = tk.Radiobutton(window, text="Prompt Writing with adversarial loop", variable=choice_var, value="2",
                             command=lambda: on_select(choice_var.get(), window))

    # 输入状态显示框
    username_entry_statu = tk.Text(window, height=1.5, width=20)
    Question_entry_statu = tk.Text(window, height=1.5, width=20)

    # 添加到布局
    username_label.grid(row=0, column=0, padx=10, pady=5)
    username_entry.grid(row=0, column=1, padx=10, pady=5)
    register_button.grid(row=1, column=0, padx=10, pady=5)
    username_entry_statu.grid(row=1, column=1, padx=10, pady=5)

    # 选定当前问题
    user_cur_question_label.grid(row=2, column=0, padx=10, pady=5)
    user_cur_question_entry.grid(row=2, column=1, padx=10, pady=5)
    question_sure_button.grid(row=3, column=0, padx=10, pady=5)
    Question_entry_statu.grid(row=3, column=1, padx=10, pady=5)

    guidance_button.grid(row=4, column=0, padx=10, pady=5)
    link_button.grid(row=4, column=1, padx=10, pady=5)

    option1.grid(row=5, column=0, padx=10, pady=5)
    normal_button.grid(row=5, column=1, padx=10, pady=5)

    option2.grid(row=6, column=0, padx=10, pady=5)
    agent_button.grid(row=6, column=1, padx=10, pady=5)


# Build Testing Information here
file_path = "StageI_experiment/result/username_entry.txt"
if os.path.exists(file_path):
    os.remove(file_path)

Dataset_Names = ['Computer_Human_Interaction', 'Personal_Financial_Literacy']
exp_condition_list = []
exp_Q_SQ_list = []
for Dataset_Name in Dataset_Names:
    Q = Get_Process_Full_Data(Dataset_Name=Dataset_Name, Print_Flag=False)
    print(f"Current Dataset is {Dataset_Name} and we are doing the reviewing")

    Draft_history_dict = {}
    Outcome_history_dict = {}
    All_Drafts_dict = {}
    All_Outcomes_dict = {}

    index = 0
    for item in Q.keys():
        index += 1
        print(f"\nWe are Processing: {item} in Dataset {Dataset_Name}!")
        conditions_temp = re.findall(r'\((.*?)\)', Q[item][[item for item in list(Q[item].keys()) if item.startswith("Question_")][0]])
        Q_SQ_temp = Q[item]
        exp_condition_list.append(conditions_temp)
        exp_Q_SQ_list.append(Q_SQ_temp)
        if index >= 2:
            if Dataset_Name == 'Personal_Financial_Literacy':
                exp_condition_list.append(conditions_temp)
                exp_Q_SQ_list.append(Q_SQ_temp)
            break

print("The final conditions: ", exp_condition_list)
# exit()

# 主窗口初始化
window = tk.Tk()
window.title("Prompt Writing Application")

setup_page1()  # 初始化页面1

# 运行主循环
window.mainloop()