# NLPGroup5

## Project Contributors:

- Anand Odbayar: Coding evaluator results and creating final presentation
- Blake Raphael: Coding evaluator results and creating final presentation
- Giovanni Evans: Coding evaluator results and creating final presentation
- Patrick Liu: Coding trainer and evaluator and creating final presentation
- Reilly Cepuritis: Coding evaluator results and creating final presentation

## Project Overview

This is the repository for NLP Group 5's group project. During this project, we will compare the performance of several out-of-the-box solutions at HuggingFace on the task of quantitative question answering in English, as detailed in the first task, subtask 3 on NumEval @ SemEval 2024 (See important links below). In particular we are handling the third subtask, quantitative question answering, which uses a multiple-choice format between two choices given a question. 

###### Important Links
[Dataset](https://drive.google.com/drive/folders/10uQI2BZrtzaUejtdqNU9Sp1h0H9zhLUE)
[Project Presentation](https://docs.google.com/presentation/d/1K4x0OJyhAfyJciX1ozsdWsNbCaIEuzyqk6jUJZNDq2g/edit?usp=sharing)
[HuggingFace](https://huggingface.co/)
[Task 1 Subtask 3 of NumEval @ SemEval 2024](https://sites.google.com/view/numeval/tasks?authuser=0)

## Implementation Explanation

#### Part 1: Getting Started

The first few blocks in the notebook are setting up for the rest of our code base. We start with a few imports and then loading our datasets. we recommend installing some base and NLP packages (most recent versions) using ```pip install jupyter torch numpy matplotlib``` and ```pip install nltk spacy transformers``` We also recommend users running into ussues install the latest transformers and scikit-learn using ```pip install datasets transformers``` and ```pip install -U scikit-learn```. We then manipulate our datasets to fit the tokenization methods we are using later on. The manipulation of our dataset occurs from Python block 3 until block 7. Here we are doing the following to preprocess our dataset: remove variant questions and changing the answer column to either a 1 or 0 (1 for correct, 0 for incorrect). This leaves us with a dataset that has 4 features: A question, choice a, choice b, and a label that is our correct answer. We then tokenize our data so we can set up for training. This set up ensures that new users do not have to adjust much if anything at all to preprocess our datasets. Just click run and go.

#### Part 2: Setting Up for Training

While running the blocks sequentially, we arrive to the sections with ```AutoModelForMultipleChoice``` and ```DataCollatorForMultipleChoice```. Both of these sections set up our models to fit with our adjusted datasets by setting them up in a multiple choice fashion. Next we set up our function to compute our evaluation metrics (In this case we are using F1 sccore to evaluate our models). 

#### Part 3: Training

The next section we come to is where we start to train our model. We start with setting up our model and providing arguments, encoded datasets from our training and evaluation, our tokenizer, our data collector, and finally we compute the evaluation metrics. Next we have a test block to ensure our trainer is working correctly, then we move on to our evaluator and formatting function. We then append each reference and prediction and evaluate how accurate our model is and finally print this number in a human readable format. We next define our ```trainAndEval``` function to combine our previous set up training with our evaluator

#### Part 4: Evaluating the Models

The next section of code blocks in sequential execution are setting up and evaluating different out-of-the-box models from Huggingface. In order here are the models trained and evaluated:

###### Models

- BERT base-uncased
- RoBERTa base-sentiment
- DistilBERT base-uncased
- BERT base-spanish
- RoBERTA base-SQUAD2
- Dynamic TinyBERT
- DistilBERT base-cased distilled SQUAD
- DistilBERT base-uncased finetuned SQUAD
- RoBERTa large-english sentiment
- Feel-it italian-sentiment

We print the results from the evalutions after 3 epochs of training. This does take considerable compute time (12+ hours).

#### Part 5: Results and Analysis

The outputs are the answer choice (1 or 0 from our earlier preprocessing) that are then evaluated for accuracy using F1 score. This is hardcoded so please do not forget to change these when you run your versions. These evaluation results are then stored in lists to more easily graph the results. The next blocks of code are bar charts of our grouped model types of Baseline, Sentiment Analysis, and SQUaD. we then graph the best from these categories in the last bar chart. See [here](https://docs.google.com/presentation/d/1K4x0OJyhAfyJciX1ozsdWsNbCaIEuzyqk6jUJZNDq2g/edit?usp=sharing) for our in class materials discussing our procedure and results.

### Final Results Analysis
Here are the expected test set performances for our models (accuracies):
- BERT base-uncased: 0.5185 (51.85%)
- RoBERTa base-sentiment: 0.4753 (47.53%)
- DistilBERT base-uncased: 0.4938 (49.38%)
- BERT base-spanish: 0.5432 (54.32%)
- RoBERTA base-SQUAD2: 0.4938 (49.38%)
- Dynamic TinyBERT: 0.5 (50%)
- DistilBERT base-cased distilled SQUAD: 0.4753 (47.53%)
- DistilBERT base-uncased finetuned SQUAD: 0.4691 (46.91%)
- RoBERTa large-english sentiment: 0.4382 (43.82%)
- Feel-it italian-sentiment: 0.4691 (46.91%)