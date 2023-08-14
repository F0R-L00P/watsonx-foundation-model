import os
import json
import time
import random
import getpass
import winsound
import numpy as np
from pandas import read_json
from collections import defaultdict
from rouge_score import rouge_scorer


from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
#####################################################################################
#####################################################################################
#####################################################################################
# evaluation function
def get_rouge_score(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    aggregate_score = defaultdict(list)

    for result, ref in zip(predictions, references):
        for key, val in scorer.score(result, ref).items():
            aggregate_score[key].append(val.fmeasure)

    scores = {}
    for key in aggregate_score:
        scores[key] = np.mean(aggregate_score[key])

    return scores

# setup API and project creds
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("WML_API")
}

project_id = os.getenv("WML_PROD_ID")

# load data
filename_test = 'data/Summarisation/test.json'
filename_train = 'data/Summarisation/train.json'

# read data
test_data = read_json(filename_test).T[["original_text", "reference_summary"]]
train_data = read_json(
    filename_train).T[["original_text", "reference_summary"]]

# check data
train_data.head()
test_data.head()

# Specified training samples
specified_train_indices = [
    "tosdr168", "tosdr308", "tosdr116", "tosdr216", "tosdr078", "tosdr354",
    "tosdr070", "tosdr208", "tosdr139", "tosdr092"#, "legalsum05", "legalsum70",
#    "tosdr146", "tosdr188", "tosdr097", "tosdr349", "tosdr058", "tosdr276",
#    "tosdr167", "tosdr002"
]

# Select specified training samples based on the indices
specified_train_samples = train_data.loc[specified_train_indices]

# call model
model_id = ModelTypes.FLAN_UL2

# setup parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 70,
    GenParams.MIN_NEW_TOKENS: 12, # decrease from 20 increased score
    GenParams.REPETITION_PENALTY: 1,
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY
}

# setup model
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id)

prompts_used = [
    "Extract the core ideas from the text below:\n", # Rouge1 score: (20)0.2393 -> (15)0.2471 -> (13)0.2520 -> (10)0.2466 
    "Condense the following text into a succinct summary:\n",# Rouge1 score: 0.2129
    "Distill the essence of the following document into a brief summary:\n", # Rouge1 score: 0.20
    "Summarize the following document, focusing on its key insights and conclusions:\n", # Rouge1 score: 0.225
    "Provide a brief and coherent summary of the essential points from the text below:\n", # Rouge1 score: 0.2047
    "Capture the main ideas and findings from the following text in a concise summary:\n", # Rouge1 score: 0.2108
    "Provide a concise summary that captures the main points of the following document:\n", # Rouge1 score: 0.17
    "Translate the following detailed text into a brief synopsis that retains the core meaning:\n", # Rouge1 score: 0.2102
    "Condense the text below into a short overview, highlighting significant thoughts and themes:\n", # Rouge1 score: 0.222
    "Distill the following text into a compact summary, preserving the primary ideas and conclusions:\n", # Rouge1 score: 0.2074
    "Create a succinct summary of the following document, emphasizing the central arguments and discoveries:\n", # Rouge1 score: 0.2163
    "Compile the main ideas from the following text into a concise overview that retains the essential points:\n", # Rouge1 score: 0.2061
    "Condense the essential information from the following text, highlighting the primary arguments and findings:\n", # Rouge1 score: 0.2167
    "Reduce the following text into a concise summary, focusing on the most relevant information and key takeaways:\n", # Rouge1 score: 0.2082
    "Synthesize the information from the following text into a compact summary that emphasizes the main ideas and conclusions:\n", # Rouge1 score: 0.2153

    # Combines high scoring prompts
    "Distill the core ideas from the text below into a concise summary, focusing on the most relevant information:\n", # Rouge1 score: 0.2120
                                                                                                                                     ##### (12|1|72)0.2701
    # Builds on high-scoring prompts
    "Synthesize the following document into a brief summary, capturing the key insights and findings:\n", # Rouge1 score: 0.2131
    # Introduces new wording
    "Translate the main points of the following text into a succinct overview that emphasizes essential themes and arguments:\n", # Rouge1 score: 0.2022
    # Varying phrasing for diversity
    "Condense the following document into a brief summary that highlights the primary thoughts and conclusions:\n", # Rouge1 score: 0.2138
    # Combining elements from existing prompts
    "Extract and reduce the following text into a compact summary, preserving the central ideas and key takeaways:\n",  # Rouge1 score: 0.2149
    # multi-step summarization
    "First, identify the main subject of the text below. Then, summarize the key arguments or insights, and conclude with the main takeaways:\n", # Rouge1 score: 0.2097
    # inquiry based summarization
    "What is the main argument of the text below? Summarize the evidence presented and the conclusions reached:\n", # Rouge1 score: 0.2171
    # audience based summarization
    "Write a summary of the text below suitable for non-expert readers, focusing on clear explanations of technical terms and concepts:\n" # Rouge1 score: 0.2157 
                                                                                                                                                        # (13)0.2609
                                                                                                                                                        # (12)0.2630/02640
                                                                                                                                                        # (12 pen:1.1)0.2685
                                                                                                                                                        # (12 pen:1.8)0.2618
                                                                                                                                                        # (12 pen:1.7)0.2618
                                                                                                                                                        # (12 pen:1.6)0.2636
                                                                                                                                                        # (12 pen:1)0.2694
                                                                                                                                                        # (12|pen:1|75)0.2696
                                                                                                                                                        # (12|pen:1|70)0.2697
                                                                                                                                                        # (12|pen:1|65)0.2693
]

# General instruction
general_instruction = prompts_used[15]#refined_prompts[0]#

instructions = []
for index, row in specified_train_samples.iterrows():
    instruction = f"""
    input: {row.original_text}\n
    output: {row.reference_summary}\n\n
    """
    instructions.append(instruction)

# Concatenate the general instruction with the specific examples
full_instruction = general_instruction + " ".join(instructions)

# Randomly (or all) sample documents from the test data to test the prompts
random_test_samples = test_data#.sample(40)
results = []

# start time for timing the generation process
start_time = time.time()

for input_text, reference_summary in zip(random_test_samples.original_text, 
                                         random_test_samples.reference_summary):
    test_instruction = f"""
    input: {input_text}\n
    output: \n\n
    """
    prompt = full_instruction + test_instruction
    results.append(model.generate_text(prompt=prompt))

# finish time for timing the generation process
finish_time = time.time()

# evaluate results
print(get_rouge_score(results, test_data.reference_summary.values))
print(f'total_time:{(finish_time - start_time)/60} minutes')
winsound.Beep(1000, 800)  # Frequency 1000 Hz, duration 800 ms

# results contain the generated summaries, and 'random_test_samples/full_sample' contains the test data
for generated_summary, original_text, reference_summary in zip(results, 
                                                               random_test_samples.original_text, 
                                                               random_test_samples.reference_summary):
    print("Original Text:")
    print(original_text[:500])
    print("\nGenerated Summary:")
    print(generated_summary)
    print("\nReference Summary:")
    print(reference_summary)
    print("\n" + "="*50 + "\n")  # Print separator