from openai import OpenAI
import configparser

def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo",
                                 temperature=0,
                                 max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content



config = configparser.ConfigParser()
# Read the configuration file
config.read('config.ini')

client = OpenAI(api_key  = config.get('KEYS', 'OPENAI_API_KEY'))
context = [ {'role':'system', 'content':"""Hello"""} ] 
prompt = "Hello, is this ChatGPT?"
context.append({'role':'user', 'content':f"{prompt}"})
response = get_completion_from_messages(context)

print(response)