import dashscope
import random
from http import HTTPStatus
from dashscope import Generation

from DataSet.dataset_processing import Get_Process_Full_Data
from scipy.stats import pearsonr, spearmanr, kendalltau

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

def build_dataset_increase_prompt(old_domens, new_data, easy=0.7):
    old_data = []
    for item in old_domens:
        temp = {"Sentence1": item[0][0], "Sentence2": item[0][1], "c-sts labels": item[2]}
        old_data.append(temp)
    new_data = {"Sentence1": new_data[0][0], "Sentence2": new_data[0][1], "c-sts labels": new_data[2]}
    prompt = f"I am in the process of constructing a dataset, aiming to enhance its richness in both samples and labels." \
             f" Would you assist me in evaluating whether the new samples I introduce contribute to the diversification " \
             f"of the current dataset?\n"

    prompt += "The criteria for assessment primarily involve: 1. Does the data encompass new behaviors? " \
              "2. Are the new labels aligned with the existing ones, and do they introduce additional diversity? " \
              "Feel free to exercise your judgment as well. " \
              "Attention, if the label doesn't vary too much or behave doesn't vary too much " \
              "(whether the variation is big is related to the threshold given below" \
              "when threshold is 1, the variation should be pretty small while when threshold is 0, the variation should be pretty big), " \
              "the new sample should not be treated as new.\n"

    prompt += f"Additionally, I require your expertise in deciding whether a sample should be incorporated based on the " \
              f"given access difficulty. Consider a scale from zero to one, where zero denotes absolute exclusion and " \
              f"one signifies unconditional inclusion. The current threshold for acquisition difficulty is set at {easy}.\n\n"

    prompt += f"Below is the existing dataset {old_data}. \n\n"
    prompt += f"Kindly review the sample ：{new_data}"

    prompt += "response in the following python dict format:{'Reason': your reason, 'Accept': use True/False to show " \
              "if we need to accept it}. You only need to give me the dict, " \
              "you should give the final judge based on the acquisition difficulty I told you before."

    return prompt

def Judge_sample(old_domens, new_data, model_name, easy=0.7):
    prompt = build_dataset_increase_prompt(old_domens, new_data, easy=easy)
    Agent="You are a useful ai and good at judge if we need to add data into dataset. "

    the_dict = call_with_messages_Qwen(prompt, Agent, model_name=model_name)["output"]["choices"][0]["message"]["content"]
    return the_dict, eval(the_dict)['Accept']

if __name__=="__main__":
    _, output = Judge_sample([[(["吃饭"],["睡觉"]),1,[1],[]], [(["吃饭"],["吃饭"]),1,[5],[]], [(["吃饭"],["做爱"]),1,[1],]],
                          [(["吃饭"],["睡觉"]),1,[1.3],[]], easy=1, model_name="qwen-plus")
    print((output))