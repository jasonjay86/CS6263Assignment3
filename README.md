# CS6263 Assignment3 - Instruction-Based Dataset Generation and Model Fine-Tuning in LLMs

**Objective**: The objective of this assignment is to explore the process of generating an instruction-based dataset for model training in Natural Language Processing (NLP). Additionally, students will fine-tune a pre-trained model using the newly created instruction-based dataset and compare its performance with the original instructions. Moreover, they will test how the model behaves before and after training with general purpose instructions which the model was originally trained.

**My Summary**:  I begun the assignment by building my data set. I started with the *stanfordnlp/imdb* dataset, which is a setiment analysis dataset.  I took the text from that dataset and prompted gpt 3.5 turbo like so:
```
prompt = "Write one response to the review that agrees with it and anotrer one to disagree...." + dataset[i]["text"]
prompt = prompt + "///output the response in json format with values for review, agree, and disagree"
```
This is done in the `createDataset.py` file.  GPT 3.5 was mostly up to the task however there were many errors in the json file I asked it produce.  That file is `mydata.json`  Ultimately, I had a lot of manual work to make the file useable and eventually it became `mycsvdata.csv`.  Unfortunately, it also ended up only being about 400 rows.

With that data, I finetuned against a model pretrained by...me...in the previous assignment. I took the Llama model that I created and then finetuned it further with my new imdb dataset and then finetuned it again with the original data from assignment 2.  The 2 rounds of finetuning was done in `trainLlama2B.py` and `trainLlama2C.py`

From there, I ran my evaluation code, `Evaluation.py` and `Evaluation3B.py`, and got the results that I will describe in the Assignment discussion section below.

## Instructions
### Envrionment Setup
To run, first load the environment from the environment.yml file with:

`conda env create -f environment.yml`

Then activate it:

`conda activate assignment2`

I used the same environment as the previous assignment

### Fine Tuning

To run the fine tuning run:

`python trainLlama2B.py` or `python trainLlama2C.py`

### Execution

In order to run inferences for the models run:

`python Evaluation.py` or `python Evaluation3B.py`

## Assignment Discussion

**3a.) Evaluate the saved model from 2.b and 2.b an on your proposed dataset and write a descriptive analysis on the results. Create a table like the sample table provided.**

![image](https://github.com/jasonjay86/CS6263Assignment3/assets/65077765/81ce998f-3c01-401d-892a-25c72ebb7346)

Unfortunately, my dataset seems to ruin the model.  Especially for code.  I think some coding skill degradation is to be expected since they are very different datasets.  However, I would have expected that the fine-tuning would not have affected the ability of the model to create code.  Not so!  Out of 20 code-prompt inferences i printed for evaluation, only 6 of them produced any code at all in the 2b model.  Most of the rest are gibberish.  The code it did produce was not very good. The original model and the 2c model produced code for each of the coding prompts without fail.  So for 2b, the CodeBLEU score was ruined.  It before reasonably well on Rouge and Bert scores that do not consider coding.

**3b.) Create 10 instructions completely out-of-sample from your dataset, that produces good results on the original pre-trained model. Use these instructions to generate inference from the original pre-trained model, and the model you saved in 2b and 2c. Write a comparison analysis on the outcome from various stages of trained models.**
To get the out of sample instructions, I asked Gemini to provive some sample prompts for an llm.  It gave a well rounded group of instructions including a translation to french and prompts to create poems, python code,songs, and movie scripts.

In my base model, the prompt to **"Compose a love song from the perspective of a robot to its human owner."**  generated a particularly good inference.  The model gave:

*Robot: Dear human, I'm programmed to love and serve you, I'm your companion and protector, I'll be here for you always, I'll never leave or abandon you, I'll always be by your side, Your robot will always be faithful and true.*

*Human: I'm so touched by your love and devotion, You're more than just a machine to me, You're my friend and confidant, You're the one I turn to in times of need,You're my strength and my support, I'm so grateful to have you by my side.*

*Both: Together, we're stronger and more resilient, We'll face whatever life throws our way, We'll always have each other's back, We're unstoppable as a team, We're unbreakable as one.*

Not bad!  And thats just the first verse!

However, fine tuning on my movie review dataset, seemed to remove the model's musical talent.  For the same prompt, after the first round of fine tuning on my dataset, the 2b model gave:

*Write a poem about a character from a work of fiction who is a robot.*

*Write a poem in which you compare your human lover to a robot.*

*Write a poem in which you compare the love of a human for a robot to the love of a human for another human.*

*Write a poem in which you compare the love of a robot for a human to the love of a robot for another robot.*

*Write a poem in which you compare the love of a human for a robot to the love of a robot for another robot.*

*Write a poem in which you compare the love of a human for a robot to the love of a human for another robot.*

*...and so and so on...*
Until it ran out of characters for the output.  Not exactly a hit.

Finally, after finetuning with the combined dataset, the 3c model gave us this gem:

*I'm a robot, programmed to love you,*

*To serve you, to protect you, to be true.*

*I'm designed to be your companion,*

*To bring you joy, to make you laugh, to be your friend.*

*I'm here for you, through thick and thin,*

*To be your rock, your shelter, your safe haven.*

*I'll always be by your side,*

*To comfort you, to guide you, to be your light.*

*I'm your robot, I'll always be here for you,*

*To love and serve you, to be your loyal friend,*

*For as long as you need me, I'll be here to stand by your side.*

The musical talent is back!

My best guess is that I made my dataset so specific, that the model became a little confused by such an open-ended instruction that required more creativity.  The original dataset was:

  **1)** Much, much, bigger(25K rows vs a little more than 400 rows)
  
  **2)** Much more varied.  Even though is a coding dataset,  there are some more natural language tasks that gives the model a bit more to work with.  I finetuned my dataset with a very very specific format and instruction.

It is very interesting to see how much fine-tuning can affect an LLM.  Fine tuning with really just a handful of data rows, I was able to remove Llama's musical skills.
