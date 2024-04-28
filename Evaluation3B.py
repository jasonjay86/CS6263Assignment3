from peft import PeftModel, PeftConfig
from datasets import load_dataset
import random
from codebleu import calc_codebleu
from rouge import Rouge
from bert_score import score
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
    )

def getOutput(tokenizer,model,testPrompt,hparam,size=0):
    input = tokenizer(testPrompt, return_tensors="pt").input_ids
    input = input.to('cuda')
    if hparam == "vanilla":
        # Generate output using vanilla decoding
        outputs = model.generate(input, max_length = 450)
    elif hparam == "topK":
        # Generate output using top-K sampling
        outputs = model.generate(input,
                                 max_length = 450,
                                 do_sample=True,
                                 top_k=size)
    elif hparam == "beam":
        # Generate output using beam search
        outputs = model.generate(input,
                                 max_length = 450,
                                 num_beams=size,
                                 early_stopping=True)
    elif hparam == "temp":
         # Generate output using temperature sampling
         outputs = model.generate(input,
                                 max_length = 450,
                                 do_sample=True,
                                 top_k=0,
                                 temperature = size)
    # Generate output at different model layers
    elif hparam == "layer":
        logits_processor = LogitsProcessorList(
        [
        MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ]
        )
        # instantiate logits processors
        logits_warper = LogitsProcessorList(
        [
            TopKLogitsWarper(50),
            TemperatureLogitsWarper(0.7),
        ]
        )

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=50)])

        torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        outputs = model._dola_decoding(
            input,
            dola_layers=[size],
            max_length = 450,
            repetition_penalty=1.2,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            # output_scores=True,
            # return_dict_in_generate=True,
        )


    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


modelList = [
    "./LlamaBase",
    "./Llama2b",
    "./Llama2c",
    # "./FTPhi2_dev",
    # "./Mistral",
]

outputType = [
    "vanilla",
    # "topK",
    # "beam",
    # "temp",
    # "layer"
]

topKsize = [
    2,
    4,
    6,
    8
]

beamsize = [
    2,
    3,
    4,
    5
]

tempSize = [
    .1,
    .25,
    .5,
    .75
]

layernum = [
    8,
    16,
    24,
]

datapath = "flytech/python-codes-25k"

# Load dataset
dataset = load_dataset("flytech/python-codes-25k", split='train')
numInputs = 25

randrows = []
referencelist = []

testPrompts = []
for i in range(numInputs):
    # randrows.append(random.randint(0,len(dataset)))
    randnum = random.randint(0,len(dataset))
    testPrompts.append(dataset[randnum]["instruction"])
    referencelist.append(dataset[randnum]["output"])

dataset = load_dataset('csv', data_files='mycsvdata.csv')
print(len(dataset['train']))
for i in range(numInputs):
    # randrows.append(random.randint(0,len(dataset)))
    randnum = random.randint(0,len(dataset['train']))
    if randnum % 2 == 0:
         text = f"### Here is a review: {dataset['train']['review'][randnum]}\n ### Tell me why you agree with the review:"
         referencelist.append(dataset['train'][randnum]["agree"])
    else:
         text = f"### Here is a review: {dataset['train']['review'][randnum]}\n ### Tell me why you disagree with the review:"
         referencelist.append(dataset['train'][randnum]["disagree"])
    testPrompts.append(text)

print(testPrompts)

for modelpath in modelList:
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(modelpath)
    model = PeftModel.from_pretrained(model, modelpath)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    for hparam in outputType:
        sizes = []
        if hparam == "vanilla":
            sizes = [1]
        elif hparam == "topK":
            sizes = topKsize.copy()
        elif hparam == "beam":
            sizes = beamsize.copy()
        elif hparam == "temp":
            sizes = tempSize.copy()
        elif hparam == "layer":
            sizes = layernum.copy()

        for size in sizes:
            predictionlist = []
            for i in range(numInputs):
                print("Getting output for: " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size) + "...Instruction:" + str(i+1))
                # testPrompt = dataset[i]["instruction"]
                
                text = getOutput(tokenizer,model,testPrompts[i],hparam,size)
                
                # referencelist.append(dataset[i]["output"])
                predictionlist.append(text)
            
            print("Results for " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size))
            print('-' * 80)
            ##codebleu##
            codebleuResult = calc_codebleu(referencelist, predictionlist, lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
            print("CodeBleu Scrore: " + str(codebleuResult["codebleu"]))
            ##rouge##
            rouge = Rouge()
            scores = rouge.get_scores(predictionlist, referencelist, avg=True)
            print("Rouge-L score: " + str(scores["rouge-l"]))
            ##BERTscore##
            P, R, F1 = score(predictionlist, referencelist, lang="en", verbose=True)
            print("BERTScore:")
            print(P, R, F1)

            print('-' * 80)
            print("")

            print("For Human Evaluation on : " + str(modelpath)+ "... output type: "+ str(hparam)+ " size = "+ str(size))
            
            #only need 20 for human evaluation
            if numInputs > 20:
                numHumanEval = 20
            else:
                numHumanEval = numInputs

            for i in range(numHumanEval):
                print("Instruction " + str(i))
                
                print(dataset[i]["instruction"])
                print("***")
                print(str(modelpath) + " output:")
                print(predictionlist[i])
                print('-' * 80)