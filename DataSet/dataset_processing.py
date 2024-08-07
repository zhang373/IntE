import os
import json
import math

def BuildDataset(Q, keyname):
    # 根据用户提供的信息，我们首先定义了一个occupations列表，其中包含不同金融素养水平的职业
    occupations = [
        # 高金融素养-5
        "金融分析师", "投资银行家", "证券交易员", "财务顾问", "审计师",
        "金融教授或讲师", "资产管理专家", "保险精算师", "企业CFO", "金融市场研究员",
        # 中等金融素养-3
        "金融销售人员", "银行柜员", "财务会计", "中小企业主", "个人理财规划师",
        "证券经纪人", "会计师", "经济学学生", "贷款专员", "风险管理专员",
        # 低金融素养-1
        "建筑工人", "清洁工", "行政人员", "零售店员", "餐厅服务员",
        "艺术家", "创意工作者", "农民", "牧民", "非营利组织员工",
        "退休人员", "非经济或商业专业学生", "自由职业者", "个体经营者", "医生",
        "律师", "工程师", "市场营销专员", "IT专家", "教师", "记者"
    ]
    """
    # 假设Q字典的结构如下所示
    Q = {
        'Question_1': {
            'Question_1': 'financial behavior)您办理银行业务的频率如何？(financial behavior)您一般在银行办理哪些业务？(financial literacy)您认为将钱放在银行会存在什么风险？',
            'questions_1': ['您办理银行业务的频率如何？', '您一般在银行办理哪些业务？', '您认为将钱放在银行会存在什么风险？'],
            'responses_chatglm': {
                '金融分析师': ['作为金融分析师，我几乎每天都会与银行进行交互，以确保我的投资组合与市场同步。', '我主要办理的业务包括投资组合的再平衡、市场数据的分析、以及利用银行平台进行的各类金融交易。', '将钱存放在银行的风险主要包括通货膨胀导致的购买力下降、银行信用风险、以及可能的流动性风险。'],
                # 其他职业的数据省略
            },
            # 其他响应数据省略
        }
    }
    """

    # 创建数据集
    dataset = []

    # 为每个职业赋予相应的金融素养标签
    occupation_to_label = {occupation: 5 for occupation in occupations[:10]}
    occupation_to_label.update({occupation: 3 for occupation in occupations[10:20]})
    occupation_to_label.update({occupation: 1 for occupation in occupations[20:]})

    # 遍历responses_chatglm字典
    for occupation, responses in Q[keyname].items():
        # print(occupation, responses)
        label = occupation_to_label.get(occupation, 1)  # 如果职业不在列表中，默认为低金融素养
        # print("Test loop")
        # for response in responses:
        # print("The response shown in BuildDataset: ",response)
        # dataset.append([response, label, []])
        dataset.append([responses, label, []])

    # 将数据集添加到Q字典中，命名为原名_dataset
    # Q[keyname+'_dataset'] = dataset

    # 输出结果查看
    # dataset
    return dataset


def GetString(list_of_tuples):
    first_elements = [t[0] for t in list_of_tuples]
    return first_elements


def GetRealLabel(list_of_tuples):
    first_elements = [t[1] for t in list_of_tuples]
    return first_elements


def GetSudoLabel(list_of_tuples):
    first_elements = [t[2] for t in list_of_tuples]
    return first_elements


def Process_Raw_Data(Q, Question_n):
    dataset = {}
    for item in Q[Question_n].keys():
        if item.startswith('res') and not item.endswith('dataset'):
            # print("Current item in Raw_data: ",item)
            dataset[item + '_dataset'] = BuildDataset(Q[Question_n], item)
            # print(dataset.keys())
    # print(dataset.keys())
    Q[Question_n].update(dataset)
    # print(Q["Question_1"].keys())
    return Q


def ShowQuestion(Q, Question_n):
    print("\n\nCurrent Raw Dataste below!")
    for item in Q[Question_n].keys():
        if item.startswith('res') and not (item.endswith('Prompt') or item.endswith('dataset')):
            print(item)
            # print(Q["Question_1"][item])
    print("\nCurrent Processed Dataste below!")
    for item in Q[Question_n].keys():
        if item.endswith('dataset'):
            print(item)
            # print(Q["Question_1"][item])
    print("\nCurrent Processed Prompt Dataste below!")
    for item in Q[Question_n].keys():
        if item.endswith('Prompt'):
            print(item)



def test_data(Q, Print_Flag=False):
    if Print_Flag:
        print(Q.keys())
        print(Q[list(Q.keys())[0]].keys())
        for index in range(len(list(Q.keys()))):
            ShowQuestion(Q, list(Q.keys())[index])
    for index in range(len(list(Q.keys()))):
        print("current Processed Q list: ", list(Q.keys())[index])
        Process_Raw_Data(Q, list(Q.keys())[index])
    if Print_Flag:
        for index in range(len(list(Q.keys()))):
            ShowQuestion(Q, list(Q.keys())[index])
        print("\n\n\n String", GetString(Q['Question_1']['responses_chatglm_dataset']))
        print("\n\n\n Real", GetRealLabel(Q['Question_1']['responses_chatglm_dataset']))
        print("\n\n\n Sudo", GetSudoLabel(Q['Question_1']['responses_chatglm_dataset']))
    return Q




def generate_pairs(strings, Process_String=True):
    # print("generate_pairs(strings)", strings)
    pairs = []
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            if Process_String:
                pairs.append([(strings[i], strings[j]), 0, [], []])
            else:
                pairs.append([(strings[i], strings[j]), 0])
    return pairs


def replace_labels_with_int_difference(pair_list):
    # Replace the placeholder label with 5 minus the absolute difference of the tuple elements
    for pair in pair_list:
        diff = abs(pair[0][0] - pair[0][1])  # Calculate the absolute difference between the two integers
        pair[1] = 5 - diff
    return pair_list


def BuildingPromptDataset(Q, Question_n):
    item_buffer = []
    data_buffer = []
    for item in Q[Question_n]:
        if item.startswith('res') and item.endswith('dataset'):
            # print("Current Processed item: ", item)
            metadata_ulabeled = GetString(Q[Question_n][item])
            # print("metadata_ulabeled: \n",(metadata_ulabeled))
            metadata_reallabel = GetRealLabel(Q[Question_n][item])
            # print("metadata_reallabeled: \n",len(metadata_reallabel))

            pair_list_reallabel = generate_pairs(metadata_reallabel, Process_String=False)
            pair_list_unlabeled = generate_pairs(metadata_ulabeled, Process_String=True)
            if (len(pair_list_reallabel) != math.comb(len(metadata_reallabel), 2)):
                raise ValueError("Current")
            # print("len(pair_list_reallabel): ", (pair_list_reallabel[0]), pair_list_unlabeled[0])

            # 示例：使用整数对列表
            pair_list_reallabel = replace_labels_with_int_difference(pair_list_reallabel)
            # print("len(pair_list_reallabel): ", (pair_list_reallabel[0]), pair_list_unlabeled[0])

            for index in range(len(pair_list_reallabel)):
                pair_list_unlabeled[index][1] = pair_list_reallabel[index][1]
                # print("len(pair_list_reallabel): ", (pair_list_reallabel[index]), pair_list_unlabeled[index])

            # print(item_buffer)
            item_buffer.append(item)
            data_buffer.append(pair_list_unlabeled)

    for index in range(len(item_buffer)):
        Q[Question_n][item_buffer[index] + '_Prompt'] = data_buffer[index]
    return Q


def Traverse_and_Retrieve(Dataset_Name='Personal_Financial_Literacy'):
    # 设定要遍历的文件夹路径
    folder_path = "G:/PhD/EMNLP/IntE/DataSet/" + Dataset_Name
    # 初始化一个列表来存储所有读取的 JSON 数据
    Q = {}
    # 遍历文件夹
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件名是否符合 "Question_x_data.json" 的格式
            if file.endswith('_data.json'):
                # 构建完整的文件路径
                file_path = os.path.join(subdir, file)
                # 读取 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    try:
                        # 加载 JSON 数据
                        data = json.load(json_file)
                        # 将读取的数据添加到列表中
                        # print("data.keys()[0]: ", (list(data.keys())[0]))
                        Q[(list(data.keys())[0])] = data[list(data.keys())[0]]
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {file_path}: {e}")
    return Q

def Get_Process_Full_Data(Dataset_Name = 'Personal_Financial_Literacy', Print_Flag=False):
    print(f"We are processing dataset{Dataset_Name} and will we print details: {Print_Flag}")
    # 指定要读取的dataset类型
    Dataset_Name = Dataset_Name
    # 从 JSON 文件中读取字典
    Q = Traverse_and_Retrieve(Dataset_Name)
    # print(Q['Question_2'].keys())
    # 打印读取的字典内容并构建数据集
    #Q = test_data(Q, Print_Flag=False)
    for index in range(len(list(Q.keys()))):
        print("current Processed Q list: ", list(Q.keys())[index])
        Process_Raw_Data(Q, list(Q.keys())[index])
        Q = BuildingPromptDataset(Q, list(Q.keys())[index])

    # Show Current Data
    if Print_Flag:
        for index in range(len(list(Q.keys()))):
            print(f"\n\nWe are showing the Question_{index+1}")
            ShowQuestion(Q, list(Q.keys())[index])
            print(Q[list(Q.keys())[index]]['responses_qwen_dataset_Prompt'][0])
    print("Dataset Q is Processed!\n")
    return Q

if __name__ == "__main__":
    Get_Process_Full_Data(Dataset_Name='Personal_Financial_Literacy', Print_Flag=False)