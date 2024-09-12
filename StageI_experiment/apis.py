import random
from http import HTTPStatus
import dashscope
from openai import AzureOpenAI


def call_gpt_ablation(content="You are a helpful assistant", agent_des="hello, nice to meet you!", model_name="gpt-4o-mini"):
    client = AzureOpenAI(
        api_key="e68059b3be9f4cf98d9a179c5a17b722",
        api_version="2024-06-01",
        azure_endpoint="https://hkust.azure-api.net"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": agent_des}
        ]
    )
    text = response.choices[0].message.content
    return text

def call_with_messages_Qwen_ablation(content, agent_des, model_name="qwen-plus"):
    dashscope.api_key = 'sk-ecb103e3471849b1b3de6cef5bde581f'
    if model_name == "qwen-turbo":
      model = dashscope.Generation.Models.qwen_turbo
    if model_name == "qwen-plus":
      model = dashscope.Generation.Models.qwen_plus
    if model_name == "qwen-max":
      model = dashscope.Generation.Models.qwen_max
    messages = [{'role': 'system', 'content': agent_des},
                {'role': 'user', 'content': content}]
    response = dashscope.Generation.call(
        model,
        messages=messages,
        seed=random.randint(1, 10000),
        result_format='message',
    )
    if response.status_code == HTTPStatus.OK:
        # print("\n\n\nJob Done!", response["output"]["choices"][0]["message"]["content"])
        return (response["output"]["choices"][0]["message"]["content"])
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

model_list_openai = ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-35-turbo", "gpt-4o-mini"]
price_list_openai = ["$30.00", "$10.00", "$5.00", "$3.00", "$0.150 "]
for model_name_openai in model_list_openai:
    print(f"Current model is: {model_name_openai}")
    print(f"Current outcome: \n {call_gpt_ablation(model_name_openai)}")
    #print("Topic Done")

model_list_tyqw = ["qwen-max", "qwen-plus", "qwen-turbo"]
for model_name_tyqw in model_list_tyqw:
    print(f"Current model is: {model_name_tyqw}")
    print(f"Current outcome:")
    outcome = call_with_messages_Qwen_ablation(content="nice to meet you", agent_des=f"You are a good assistant!", model_name=model_name_tyqw)
    print(outcome)

print("\n\n\nhello wenshuo, we have finished the api calling")