import random
from http import HTTPStatus
import dashscope
from DataSet.dataset_processing import Get_Process_Full_Data
from scipy.stats import pearsonr, spearmanr, kendalltau
from Robort2 import *


def Test_Connection():
    try:
        Out_Qwen = call_with_messages_Qwen(model_name="qwen-turbo",
                                           content='你现在扮演一个小学生，不太懂做饭,你只能根据你当前已有的知识做出回答。问题：怎么做鱼？',
                                           agent_des="你是一个小学生，不太懂做饭")
        print(Out_Qwen["output"]["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error Happened and Error type is: {e}. You need to retry the connection between server and local host!")

def Cal_Q(Q):
    # 遍历Q字典中的每个Question
    for question_key in Q:
        #print("The question_key: ", Q[question_key].keys(), Q[question_key]['Question_1'], Q[question_key]['questions_1'])
        Current_Question = Q[question_key][[item for item in list(Q[question_key]) if item.startswith("questions_")][0]]
        print(f"We are processing the question to cal sodu label: {Current_Question}")
        # 遍历每个以'_Prompt'结尾的数据项
        for prompt_key in filter(lambda k: k.endswith('_Prompt'), Q[question_key].keys()):
            print("The prompt_key: ", prompt_key)
            # 遍历每个响应列表，并计算相似性
            for response in Q[question_key][prompt_key]:
                if len(response) == 4 and not response[2]:  # 确保第三个元素是空列表
                    # 计算相似性并存储结果
                    if isinstance(Cal_sim(response[0], Current_Question), list):
                        response[2].extend(Cal_sim(response[0], Current_Question))
                    else:
                        response[2].append(Cal_sim(response[0], Current_Question))
    # 打印修改后的Q字典，以验证结果
    print("We have finish C-STS Cal! Please Check Q!\n")
    return Q

class ListSimilarity:
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2

    def hamming_distance(self):
        return sum(el1 != el2 for el1, el2 in zip(self.list1, self.list2))

    def euclidean_distance(self):
        return sum((el1 - el2) ** 2 for el1, el2 in zip(self.list1, self.list2)) ** 0.5

    def manhattan_distance(self):
        return sum(abs(el1 - el2) for el1, el2 in zip(self.list1, self.list2))

    def cosine_similarity(self):
        dot_product = sum(el1 * el2 for el1, el2 in zip(self.list1, self.list2))
        magnitude_list1 = sum(el1 ** 2 for el1 in self.list1) ** 0.5
        magnitude_list2 = sum(el2 ** 2 for el2 in self.list2) ** 0.5
        return dot_product / (magnitude_list1 * magnitude_list2)

    def jaccard_similarity(self):
        set1 = set(self.list1)
        set2 = set(self.list2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    def pearson_correlation(self):
        mean_list1 = sum(self.list1) / len(self.list1)
        mean_list2 = sum(self.list2) / len(self.list2)
        numerator = sum((el1 - mean_list1) * (el2 - mean_list2) for el1, el2 in zip(self.list1, self.list2))
        denominator = (sum((el1 - mean_list1) ** 2 for el1 in self.list1) *
                       sum((el2 - mean_list2) ** 2 for el2 in self.list2)) ** 0.5
        return numerator / denominator if denominator != 0 else 0

    def spearman_correlation(self):
        rank_list1 = sorted(range(len(self.list1)), key=lambda k: self.list1[k])
        rank_list2 = sorted(range(len(self.list2)), key=lambda k: self.list2[k])
        d = [abs(rank_list1[i] - rank_list2[i]) for i in range(len(self.list1))]
        d_squared = sum([di ** 2 for di in d])
        n = len(self.list1)
        return 1 - (6 * d_squared) / (n * (n**2 - 1))

    def kendalls_tau(self):
        n = len(self.list1)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                concordant += (self.list1[i] - self.list1[j]) * (self.list2[i] - self.list2[j]) > 0
                discordant += (self.list1[i] - self.list1[j]) * (self.list2[i] - self.list2[j]) < 0
        return (concordant - discordant) / (0.5 * n * (n - 1))

    def get_all_similarities(self):
        return {
            "Hamming Distance": self.hamming_distance(),
            "Euclidean Distance": self.euclidean_distance(),
            "Manhattan Distance": self.manhattan_distance(),
            "Cosine Similarity": self.cosine_similarity(),
            "Jaccard Similarity": self.jaccard_similarity(),
            "Pearson Correlation": self.pearson_correlation(),
            "Spearman Correlation": self.spearman_correlation(),
            "Kendall's Tau": self.kendalls_tau()
        }


def cal_other_eval(list1, list2):
    # Creating an instance of ListSimilarity and testing
    similarity_checker = ListSimilarity(list1, list2)
    results = similarity_checker.get_all_similarities()
    # print(results)
    return results


def calculate_correlations(response1, response2):
    # 确保两个列表的长度相同
    if len(response1) != len(response2):
        raise ValueError("The lists must have the same length.")
    # 初始化相关性的结果字典
    correlations = {
        'pearson': None,
        'spearman': None,
        'kendall': None
    }

    # 计算Pearson相关性
    correlations['pearson'], _ = pearsonr(response1, response2)
    # 计算Spearman相关性
    correlations['spearman'], _ = spearmanr(response1, response2)
    # 计算Kendall's tau相关性
    correlations['kendall'], _ = kendalltau(response1, response2)
    return correlations

def Cal_Evla(Q):
    # 遍历Q字典中的每个Question
    for question_key in Q:
        print(f"We are process {question_key} to cal corr")
        Question_prompt_name_list=[]
        Question_prompt_corr_list = []
        # 遍历每个以'_Prompt'结尾的数据项
        for prompt_key in filter(lambda k: k.endswith('_Prompt'), Q[question_key].keys()):
            print("The prompt_key: ", prompt_key)
            Real_label = []
            Sudo_label = []
            for response in Q[question_key][prompt_key]:
                # real label in response[1], sudo label in response[2], response[3] is empty for future usage
                if len(response) == 4 and response[2]:  # 确保第三个元素不是空列表
                    Real_label.append(response[1])
                    Sudo_label.append(response[2])

            corr_list = []
            for index in range(len(Sudo_label[0])):
                temp_dict = {} #calculate_correlations(Real_label, [sublist[index] for sublist in Sudo_label])
                temp_dict.update(cal_other_eval(Real_label, [sublist[index] for sublist in Sudo_label]))
                corr_list.append(temp_dict)
            Question_prompt_name_list.append(prompt_key+"_corr")
            Question_prompt_corr_list.append(corr_list)
        for name, corr in zip(Question_prompt_name_list,Question_prompt_corr_list):
            Q[question_key][name] = corr
        break
    # 打印修改后的Q字典，以验证结果
    print("We have finish Eval on Current Dataset! Please Check Q!\n")
    return Q

if __name__ == "__main__":
    Test_Flag = False
    if Test_Flag:
        Test_Connection()

    print("We are processing Q in LLM Api demo!")
    Q = Get_Process_Full_Data(Dataset_Name='Personal_Financial_Literacy', Print_Flag=False)

    # Cal sudo label by tuned LLM Agent
    Q = Cal_Q(Q)
    print("We have finish the cal and result example is shown here: \n", Q['Question_1']['responses_chatglm_dataset_Prompt'][0],"\n")

    # Cal Corr between sudo label and real label
    Q = Cal_Evla(Q)
    print("Current Process Eval Result")