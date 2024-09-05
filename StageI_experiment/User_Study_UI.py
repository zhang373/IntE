import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import webbrowser
import os
import re
import shutil

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
def open_link():
    webbrowser.open("https://wj.qq.com/s2/14975363/pvp7/")

# 用于处理选项点击的函数
def on_select(value, the_windows):
    if value == "1":
        show_page(2, the_windows)
    elif value == "2":
        show_page(3, the_windows)

# 用于显示特定页面的函数
def show_page(page_number, the_windows):
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

# 设置Page1
def setup_page1():
    global username_entry, choice_var

    # 用户名输入
    username_label = tk.Label(window, text="Enter your username:")
    username_entry = tk.Entry(window)
    register_button = tk.Button(window, text="Register user entry", command=register_user)

    # 链接按钮
    link_button = tk.Button(window, text="Visit Questionnaire Website", command=open_link)

    # 选项
    choice_var = tk.StringVar(value="1")
    option1 = tk.Radiobutton(window, text="Prompt Writing with traditional GPT", variable=choice_var, value="1",
                             command=lambda: on_select(choice_var.get(), window))
    option2 = tk.Radiobutton(window, text="Prompt Writing with adversarial loop", variable=choice_var, value="2",
                             command=lambda: on_select(choice_var.get(), window))

    # 添加到布局
    username_label.grid(row=0, column=0, padx=10, pady=5)
    username_entry.grid(row=0, column=1, padx=10, pady=5)
    register_button.grid(row=1, columnspan=2, padx=10, pady=5)
    link_button.grid(row=2, columnspan=2, padx=10, pady=5)
    option1.grid(row=3, column=0, columnspan=2, padx=10, pady=5)
    option2.grid(row=4, column=0, columnspan=2, padx=10, pady=5)


# Build Testing Information here
file_path = "StageI_experiment/result/username_entry.txt"
if os.path.exists(file_path):
    os.remove(file_path)

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
        Q_SQ = Q[item]
        break

# 主窗口初始化
window = tk.Tk()
window.title("Prompt Writing Application")

setup_page1()  # 初始化页面1

# 运行主循环
window.mainloop()