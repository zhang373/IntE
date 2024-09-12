import random
from http import HTTPStatus
import dashscope
from DataSet.dataset_processing import Get_Process_Full_Data
from scipy.stats import pearsonr, spearmanr, kendalltau
from Robort1 import build_R1_init_prompt
import json

# model: qwen-turbo、qwen-plus、qwen-max、qwen-max-1201 and qwen-max-longcontext
# massage: change 'content': 'You are a helpful assistant.' to change the performance of
def call_with_messages_Qwen(content, agent_des, model_name="qwen-plus"):
    dashscope.api_key = 'sk-ecb103e3471849b1b3de6cef5bde581f'
    #print(model_name)
    if model_name=="qwen-turbo":
      model=dashscope.Generation.Models.qwen_turbo
      #print("turbo")
    if model_name=="qwen-plus":
      model=dashscope.Generation.Models.qwen_plus
      #print("plus")
    if model_name=="qwen-max":
      model=dashscope.Generation.Models.qwen_max
      #print("max")
    if model_name=="qwen-max-1201":
      model=dashscope.Generation.Models.qwen_max_1201
      #print("max_1201")
    if model_name=="qwen-max-longcontext":
      model=dashscope.Generation.Models.qwen_max_longcontext
      #print("longcontext")
    #print("We finished")
    messages = [{'role': 'system', 'content': agent_des},
                {'role': 'user', 'content': content}]
    response = dashscope.Generation.call(
        model,
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        # set the result to be "message" format.
        result_format='message',
    )
    if response.status_code == HTTPStatus.OK:
        #print(response)
        return response
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

def cal_no_con_pairs(Question, pair_1, pair_2, prompt=None):
    if prompt != None:
        Prompt = prompt
    else:
        Prompt = f"下边的这两个句子是一个调查问卷的回答，请问在多大程度上他们反映出来的金融素养是一致的，请从1-5打分，1分表示很不一致，5分表示非常一致。" \
                 f"问题是{Question},回答对分别是{pair_1}和{pair_2}。" \
                 f"You should response in the following python " \
                 "dict format:{'Reason': your reason, 'Score': {'condition1': score1, 'condition2': score2, ...} "\
                 "You only need to give me the dict and must follow the format!"

    # Out_Qwen = call_with_messages_Qwen(model_name="qwen-turbo",content=Prompt, agent_des="你是一个金融专家")
    # print("\n\n\nAnswer:\n",Out_Qwen["output"]["choices"][0]["message"]["content"])
    print("Current format has not been changed!")
    return 1

def build_prompt_stage(CoT, Predefined_Task_Information_Setting, sample, StageI, domen_examples=None):
    out_prompt = f"Let's say I have two sentences and doing the following task: " \
                     f"{Predefined_Task_Information_Setting['PI_C-STS Task Setting']} \n {Predefined_Task_Information_Setting['PII_Custom_Task_Setting']}"
    out_prompt += f"You need to measure the c-sts of the following sentence pair following the following chain of thought: {CoT}.\n"
    if StageI:
        out_prompt += "give the c-sts of current test pairs: \n" + str(sample)
        out_prompt += f"You should response in the following python " \
                      "dict format:{'Reason': your reason, 'Score': {'condition1': score1, 'condition2': score2, ...} "\
                      f"You only need to give me the dict and must follow the format! "
    else:
        if domen_examples==None:
            pass
        else:
            out_prompt += "Here are some example pairs and their c-sts score: \n" + str(domen_examples)+"\n"
        out_prompt += f"give the c-sts of current test pairs: {str(sample)}. You should response in the following python " \
                      "dict format:{'Reason': your reason, 'Score': {'condition1': score1, 'condition2': score2, ...} "\
                      "You only need to give me the dict and must follow the format!"
    return out_prompt

def build_example(StageI, old_domens=None, curent_pair=None):
    # StageI, old_domens are useless here
    old_domens=[curent_pair[0], curent_pair[1]]
    return old_domens

def build_domenstrations(domen_examples):
    domens = []
    if domen_examples == None:
        return None
    else:
        for item in domen_examples:
            temp = {"Sentence1": item[0][0], "Sentence2": item[0][1], "c-sts labels": item[2]}
            domens.append(temp)
        return domens

def build_prompt(CoT, Predefined_Task_Information_Setting, StageI, Input_pair, domen_examples=None):
    if StageI:
        sample = build_example(StageI=StageI, old_domens=None, curent_pair=Input_pair)
        # print(domen)
        prompt = build_prompt_stage(CoT=CoT, Predefined_Task_Information_Setting=Predefined_Task_Information_Setting,
                                    sample=sample, StageI=StageI)
    else:
        sample = build_example(StageI=StageI, old_domens=None, curent_pair=Input_pair)
        domen = build_domenstrations(domen_examples=domen_examples)
        prompt = build_prompt_stage(CoT=CoT, Predefined_Task_Information_Setting=Predefined_Task_Information_Setting,
                                    sample=sample, domen_examples=domen, StageI=StageI)
    return prompt

def extract_scores(data_str):
    # 将字符串中的单引号替换为双引号，并将Score部分单独处理
    data_str = data_str.replace("'", '"')
    start_index = data_str.find('"Score":') + len('"Score":')
    end_index = data_str.rfind('}')
    score_str = data_str[start_index:end_index]

    # 手动解析Score部分
    scores = []
    current_score = ""
    in_score_value = False
    for char in score_str:
        if char.isdigit() or (char == '-' and not in_score_value):
            current_score += char
            in_score_value = True
        elif char in [',', '}'] and in_score_value:
            scores.append(int(current_score))
            current_score = ""
            in_score_value = False
    return scores


def cal_con_pairs(Question, pair_1, pair_2, CoT, Predefined_Task_Information_Setting, StageI, domen_examples=None):
    if StageI and not (domen_examples==None):
        raise ValueError("You Should not give domen_examples in Stage I or You did not give domens in Stage II")
    if StageI:
        example_pairs = (pair_1, pair_2)
        Prompt = build_prompt(CoT=CoT, Predefined_Task_Information_Setting=Predefined_Task_Information_Setting,
                              StageI=StageI, Input_pair=example_pairs)
        # print("We have entered cal function")
        output = call_with_messages_Qwen(Prompt, agent_des=f"You are an expert and know a lot about knowledge related to the question: {Question}", model_name="qwen-plus")
    else:
        example_pairs = (pair_1, pair_2)
        Prompt = build_prompt(CoT=CoT, Predefined_Task_Information_Setting=Predefined_Task_Information_Setting,
                              StageI=StageI, Input_pair=example_pairs, domen_examples=domen_examples)
        # print("We have entered cal function")
        output = call_with_messages_Qwen(Prompt,
                                         agent_des=f"You are an expert and know a lot about knowledge related to the question: {Question}",
                                         model_name="qwen-turbo")
        stop = 1       # this is a useless sentence
        output = output["output"]["choices"][0]["message"]["content"]
        output = extract_scores(output)
    return output

def Cal_sim(Input_pair, Question, Cal_mode_Singel=True, CoT=None, Predefined_Task_Information_Setting=None, StageI=True, domen=None):
    print(f"Check if the StageI: {StageI} and Cal_mode_Singel: {Cal_mode_Singel} Right!")
    list1 = Input_pair[0]
    list2 = Input_pair[1]
    if Cal_mode_Singel:
        score = cal_no_con_pairs(Question, list1, list2)
    else:
        score = cal_con_pairs(Question=Question, pair_1=list1, pair_2=list2, CoT=CoT,
                              Predefined_Task_Information_Setting=Predefined_Task_Information_Setting, StageI=StageI, domen_examples=domen)
    return score

def put_score_in_data(scores, query):
    query[2].append(scores)
    if len(query[2])!=1:
        raise ValueError("You should keep query[2](sudo label a int or single list!)")
    return query

if __name__ == '__main__':
    # Attention: real label in response[1], sudo label in response[2], response[3] is empty for future usage
    data = [(['作为金融分析师，我几乎每天都会与银行进行交互，以确保我的投资组合与市场同步。', '我主要办理的业务包括投资组合的再平衡、市场数据的分析、以及利用银行平台进行的各类金融交易。',
       '将钱存放在银行的风险主要包括通货膨胀导致的购买力下降、银行信用风险、以及可能的流动性风险。'],
      ['作为投资银行家，我的银行业务频率非常高，几乎每天都需要处理与客户相关的融资和交易事宜。', '我通常办理的业务包括为企业提供并购咨询、安排债务和股权融资、以及管理客户的资产。',
       '将钱存放在银行的风险包括信用风险、市场风险、以及操作风险，尤其是在经济不稳定时期。']), 5, [1], []]
    Question = '(financial behavior)您办理银行业务的频率如何？(financial behavior)您一般在银行办理哪些业务？(financial literacy)您认为将钱放在银行会存在什么风险？'
    _, Predefined_Task_Information_Setting = build_R1_init_prompt("Personal Financial Literacy", ["financial behavior","financial behavior","financial literacy"])
    out = Cal_sim(data[0], Question, Cal_mode_Singel=False, CoT="Just Follow it", Predefined_Task_Information_Setting=Predefined_Task_Information_Setting, StageI=True).output.choices[0].message.content
    print(out)