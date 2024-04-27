from openai import OpenAI
from datasets import load_dataset
import json

def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo-0125",
                                 temperature=.2,
                                 max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
dataset = load_dataset("stanfordnlp/imdb", split='train')

# review = dataset[16]["text"]
# print(review)
client = OpenAI()
response =""

for i in range(len(dataset)):
  response=""
  context = [ {'role':'system', 'content':"""You are a movie critic"""} ] 
  prompt = "Write one response to the review that agrees with it and anotrer one to disagree...." + dataset[i]["text"]
  prompt =prompt + "///output the response in json format with values for review, agree, and disagree"
  context.append( {'role':'user', 'content':f"{prompt}"})
  response = response + get_completion_from_messages(context)
  if i < len(dataset):
     response = response + ",\n"
  f = open("mydata.json", "a")
  f.write(response)
  f.close()