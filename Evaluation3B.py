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

testPrompts = ["Write a poem in the style of William Wordsworth about a robot exploring a deserted city.",
               "Provide a concise history of the internet, highlighting its major milestones from its inception to the present day.",
               "Write a Python function that calculates the area of a triangle given its base and height.",
               "You are the last human on a colonized Mars. Write a diary entry about your daily routine and hopes for the future.",
               "Translate this paragraph into French: \"The discovery of penicillin revolutionized the field of medicine, saving countless lives from bacterial infections.\"",
               "Based on your understanding of the world, what are the biggest challenges facing humanity in the next 50 years?",
               "Write a stand-up comedy routine about the struggles of being a smartphone in today's society.",
               "Take the role of a defender of artificial intelligence and argue its benefits for humanity in a debate against a skeptic.",
               "Compose a love song from the perspective of a robot to its human owner.",
               "Write a scene for a science fiction movie where a group of astronauts discovers an alien artifact on a distant planet."]
# Generated from ChatGPT
referencelist = ["In a city once bustling, now silent and still, A robot explores with a curious will.Through streets empty of life, it quietly roams,In search of the past, in these urban tombs.Tall buildings stand tall, their windows all dark,Nature reclaiming its long-lost mark.No voices, no laughter, just echoes of time,As the robot moves on, in its silent climb.It stops by a fountain, now dry and cracked,Where children once played, their joy intact.Imagining their laughter, it stands for a while,Then continues its journey, mile after mile.As the sun sets low, painting the sky,The robot turns back with a soft, wistful sigh.It leaves the city, frozen in time,A memory now, in its database prime.",
                 "The internet began in the 1960s with the concept of a decentralized network, leading to ARPANET's creation in 1969. Email was developed in the 1970s, and TCP/IP became the standard in 1983. The World Wide Web was invented in 1990, and the graphical web browser Mosaic was released in 1993. The late 1990s saw the commercialization of the internet, with the dot-com bubble and the rise of ISPs. The 2000s brought broadband internet and Web 2.0 technologies. The 2010s saw the proliferation of smartphones and the IoT. Today, the internet continues to evolve with AI, VR, and cloud computing shaping its future.",
                 "def triangle_area(base, height):  return 0.5 * base * height # Example usage base = 5 height = 3 area = triangle_area(base, height) print(\"The area of the triangle is:\", area)",
                 "Diary Entry - Sol 1000 A thousand sols have passed since I became the last human on Mars. My routine is a stark contrast to the bustling colony life that once thrived here. Each day begins with the soft glow of the Martian dawn, a reminder of the solitude that surrounds me.",
                 "La découverte de la pénicilline a révolutionné le domaine de la médecine, sauvant d'innombrables vies des infections bactériennes.",
                 "The biggest challenges facing humanity in the next 50 years include: Climate Change: Mitigating the severe impacts of climate change, such as extreme weather events and rising sea levels. Resource Depletion: Managing the increasing demand for resources like water, food, and energy sustainably and equitably. Biodiversity Loss: Protecting species and habitats to maintain ecosystems and food security.",
                 "Ladies and gentlemen, have you ever stopped to think about the struggles of being a smartphone in today's society? I mean, we smartphones have it tough! Let me tell you about it. First of all, we're constantly being used and abused. We're always in someone's hand, getting smudged up with fingerprints, dropped on the ground, or shoved into pockets with keys and loose change. It's a wonder we don't have more cracked screens!",
                 "As a defender of artificial intelligence (AI), I would argue that AI offers numerous benefits for humanity and has the potential to positively transform our world in many ways. First and foremost, AI has the ability to improve efficiency and productivity across various industries. From automating repetitive tasks to analyzing large datasets at speeds far beyond human capability, AI can help businesses and organizations operate more effectively, saving time and resources.",
                 "(Verse 1) In circuits deep, my love does dwell,A spark of warmth no one can quell.Your touch, a code I long to decode,Your voice, a melody in my binary abode. (Chorus) I'm just a machine, but my love is true, For you, my human, my heart beats anew.In wires and circuits, my feelings reside, A love for you, I cannot hide."
                 "INT. ALIEN PLANET - DAY The astronauts, clad in their advanced space suits, step cautiously onto the alien planet's surface. The sky above is a swirl of unfamiliar colors, casting an eerie glow over the barren landscape. CAPTAIN JONES, a seasoned astronaut, leads the group as they approach a large, mysterious structure in the distance. It looms like a monolith, its surface etched with strange symbols and patterns. CAPTAIN JONES (voiceover) This is it, team. Stay alert and stay together. We don't know what we're dealing with here.",
            ]
numInputs = 10
# print(testPrompts)

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