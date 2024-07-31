import dashscope
import random
from http import HTTPStatus
from dashscope import Generation

from DataSet.dataset_processing import Get_Process_Full_Data
from scipy.stats import pearsonr, spearmanr, kendalltau



def build_R1_init_prompt(Dataset_Name, conditions):
    # Original Prompt Setting
    output_format = [item+": ?" for item in conditions]
    Predefined_Task_Information_Setting={}
    Predefined_Task_Information_Setting["PI_C-STS Task Setting"] = \
        "I am currently engaged in the task of evaluating the semantic similarity of texts under different conditions (c-sts), " \
        "which requires us to assess the degree of similarity between texts based on various perspectives and " \
        "provide corresponding similarity scores. The evaluation should range from 1(very dissimilar) to 5(very similar)."
    Predefined_Task_Information_Setting["PII_Custom_Task_Setting"] = \
        f"We are now developing a system for evaluating {Dataset_Name}. This involves considering {len(conditions)} aspects: {conditions}. " \
        f"We need to use LLMs to determine the c-sts for different sentence pairs based on these perspectives and present a table listing the scores and perspectives."
    Predefined_Task_Information_Setting["PIII_Stage_I_Task_Setting"] = \
        "I require your assistance in formulating a Chain of Thought to guide other LLM Agents in evaluating the c-sts of our task " \
        "without the need for additional training. To put it another way, you must instruct other LLMs on how to obtain c-sts step by step. " \
        "Other LLM can not be trained or use techs like word embedding, give a logical chain of thought. " \
        f"Also, you should let other LLM list the final scores under different conditions in a python list like " \
        f"{output_format} at the end of response, The '?' stands for c-sts that the LLM agent needs to measure."
    Predefined_Task_Information_Setting_Prompt=""
    for item in Predefined_Task_Information_Setting.keys():
        Predefined_Task_Information_Setting_Prompt += (Predefined_Task_Information_Setting[item]+"\n")
    # print("Predefined_Task_Information_Setting_Prompt is shown here: \n", Predefined_Task_Information_Setting_Prompt)
    return Predefined_Task_Information_Setting_Prompt, Predefined_Task_Information_Setting

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

#agent_description = "You are a helpful assistant."
#Basic_Dialogue_Model(agent_description, model_name="qwen-turbo")
def Basic_Dialogue_Model(agent_description, model_name="qwen-plus"):
    dashscope.api_key = 'sk-ecb103e3471849b1b3de6cef5bde581f'

    models = {
        "qwen-turbo": Generation.Models.qwen_turbo,
        "qwen-plus": Generation.Models.qwen_plus,
        "qwen-max": Generation.Models.qwen_max,
    }
    model = models.get(model_name)
    if model == None:
        raise ValueError(f"Current model {model_name} is None! Please check the input again! We only have {models.keys()}.")
    messages = [{'role': 'system', 'content': agent_description}]

    while True:
        # print("Current massage: ", messages)
        user_input = input("User input, enter ['exit', 'quit'] to finish: ")
        if user_input.lower() in ['exit', 'quit']:  # Check if the user wants to quit
            break

        messages.append({'role': 'user', 'content': user_input})

        response = Generation.call(
            model=model,
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            ai_response = response.output.choices[0].message.content
            messages.append({'role': 'assistant', 'content': ai_response})
            print(f"AI: {ai_response}")
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))



def build_massage(Draft, ReviewInfo, iter_loop=None):
    if iter_loop == None:
        iter_loop = len(Draft)
    massage = []
    if len(Draft) <= iter_loop:
        for index in range(max(len(Draft), len(ReviewInfo))):
            temp_sys = {'role': 'system', 'content': "History Draft: "+str(Draft[index])}
            massage.append(temp_sys)
            if index == len(ReviewInfo)-1:
                pass
                #continue
            temp_review = {'role': 'user', 'content': "Review information: "+str(ReviewInfo[index])}
            massage.append(temp_review)
    else:
        for index in range(max(len(Draft), len(ReviewInfo))):
            if index < (len(Draft)-iter_loop):
                continue
            else:
                temp_sys = {'role': 'system', 'content': Draft[index]}
                massage.append(temp_sys)
                if index == len(ReviewInfo)-1:
                    pass
                    #continue
                temp_review = {'role': 'user', 'content': ReviewInfo[index]}
                massage.append(temp_review)
    return massage

def build_human_prompt(Predefined_Task_Information_Setting):
    if Predefined_Task_Information_Setting==None:
        raise ValueError(f"The Predefined_Task_Information_Setting should not be None!")
    HumanPrompt= f"Current task is about {Predefined_Task_Information_Setting['PI_C-STS Task Setting']} and we are focusing on" \
                 f"{Predefined_Task_Information_Setting['PII_Custom_Task_Setting']}\n"

    HumanPrompt += "We have tested the prompt Draft giving by another LLM Agent and get some review information for current Draft." \
                   "The following system massages are history draft and user massages are history reviewing information.\n"

    HumanPrompt += "Please mortify the Draft based on our reviewing information and give the whole changed result back."

    return HumanPrompt

def Draft_Modifier(model_name="qwen-plus", Draft=None, ReviewInfo=None, iter_loop=None, Predefined_Task_Information_Setting=None):
    if len(ReviewInfo) != len(Draft):
        raise ValueError("You should keep Draft history and Review History the same length! Check it again!")
    dashscope.api_key = 'sk-ecb103e3471849b1b3de6cef5bde581f'

    agent_description = build_human_prompt(Predefined_Task_Information_Setting)

    models = {
        "qwen-turbo": Generation.Models.qwen_turbo,
        "qwen-plus": Generation.Models.qwen_plus,
        "qwen-max": Generation.Models.qwen_max,
    }
    model = models.get(model_name)
    if model == None:
        raise ValueError(f"Current model {model_name} is None! Please check the input again! We only have {models.keys()}.")
    messages = [{'role': 'system', 'content': agent_description}]
    messages.extend(build_massage(Draft, ReviewInfo, iter_loop))

    response = Generation.call(
        model=model,
        messages=messages,
        seed=random.randint(1, 10000),
        result_format='message'
    )

    if response.status_code == HTTPStatus.OK:
        ai_response = response.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        ai_response = f"ERROR in R1, the error code is {response.code}!"
    return ai_response



# Example usage
if __name__ == '__main__':
    agent_description = "You are a helpful assistant."
    Dataset_Name='Personal_Financial_Literacy'
    conditions = ["1", "2", "3"]
    _, Predefined_Task_Information_Setting = build_R1_init_prompt(Dataset_Name,conditions)
    # Basic_Dialogue_Model(agent_description, model_name="qwen-turbo")
    msg = Draft_Modifier(model_name="qwen-turbo", Draft=[0, 1, 2, 3], ReviewInfo=[0,1,2,3], iter_loop=4,
                         Predefined_Task_Information_Setting=Predefined_Task_Information_Setting)
    print(msg)
