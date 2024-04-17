# NLPGroup5

## Project Contributors:

- Anand Odbayar: Contributor (Technical Reports, Final Presentation)
- Blake Raphael: Core Contributor (Dataset Acquisition, Evaluating, Technical Reports, Final Presentation)
- Giovanni Evans: Contributor (Technical Reports, Final Presentation)
- Patrick Liu: Lead (Dataset Acquisition and Maniupulation, Training, Evaluating, Results Gathering, Technical Reports, Final Presentaion)
- Reilly Cepuritis: Contributor (Technical Reports, Final Presentation)

### Further Breakdown:

- Anand Odbayar: Anand helped contribute towards the final presentation. He created the Procedure and Analysis slides. 
- Blake Raphael: Blake contributed by finding three of our models that we tested, specifically Dynamic TinyBERT, DistilBERT Base Cased SQUaD, and DistilBERT Base Uncased SQUaD. He also wrote up a bulk of our progress report including elaborating a bit on our methodology, finding three of our related works, and explaining our next steps in the process. He also created the Premise and Hypothesis Slides as well as proofreading. Blake also helped proofread the project abstract along with finding the premise and dataset.
- Giovanni Evans: Giovanni Evans contributed towards the project abstract, progress report, and final presentation. Giovanni was the proofreader for the project abstract and progress report and contributed to the final presentation by helping create and discuss the Related Works slides. He also contributed to the final README. 
- Patrick Liu: Patrick contributed by implementing most of the code base and debugging. Patrick also helped by finding the premise and dataset along with writing up the project abstract. He found the bulk of the models to test and helped format results both in the code and on the slides. Patrick found the following models to test:  BERT base-uncased, RoBERTa base-sentiment, DistilBERT base-uncased, BERT base-spanish, RoBERTA base-SQUAD2, RoBERTa large-english sentiment, and Feel-it italian-sentiment. Patrick also created the Results section of our slides and contributed to the progress report by introducing our concept, discussing our dataset, outlining a bit of our methodology, finding a related work, and explaining a few of our next steps.
- Reilly Cepuritis: Reilly contributed by helping create the slides and proofreading.

##### Terminology:

The roles are defined as below:

- Lead: Individual(s) responsible for the entire workstream throughout the project.
- Core Contributor: Individual that had significant contributions to the workstream throughout the project.
- Contributor: Individual that had contributions to the project and was partially involved with the effort.

Other Terminology:

- Technical Reports: Project Abstract and Progress Report (Including those who wrote and proofread).


## Project Overview

This is the repository for NLP Group 5's group project. During this project, we will compare the performance of several out-of-the-box solutions at HuggingFace on the task of quantitative question answering in English, as detailed in the first task, subtask 3 on NumEval @ SemEval 2024 (See important links below). In particular we are handling the third subtask, quantitative question answering, which uses a multiple-choice format between two choices given a question. 

###### Important Links
[Dataset](https://drive.google.com/drive/folders/10uQI2BZrtzaUejtdqNU9Sp1h0H9zhLUE)

[Project Presentation](https://docs.google.com/presentation/d/1K4x0OJyhAfyJciX1ozsdWsNbCaIEuzyqk6jUJZNDq2g/edit?usp=sharing)

[HuggingFace](https://huggingface.co/)

[Task 1 Subtask 3 of NumEval @ SemEval 2024](https://sites.google.com/view/numeval/tasks?authuser=0)

## Implementation Explanation

Overall, this notebook is designed to be run sequentially. If you start from the top down, execution should be straightforward and the models should be trained and evaluated as expected.

#### Part 1: Getting Started

First we recommend grabbing the dataset for training, validation, and testing. These are found [here](https://drive.google.com/drive/folders/10uQI2BZrtzaUejtdqNU9Sp1h0H9zhLUE). Please set up your file directory the following way for the code to work: Create a "Project" folder in your repository, then create a "QQA_Data" folder within this "Project" folder. Place the dataset files within the "QQA_Data" folder. Your notebook should be outside of the parent "Project" folder for the datasets to be imported correctly. We then recommend installing some base and NLP packages (most recent versions) using ```pip install jupyter torch numpy matplotlib``` and ```pip install nltk spacy transformers``` We also recommend users running into issues install the latest datasets, transformers (At least version 4.11.0), and scikit-learn using ```pip install datasets transformers``` and ```pip install -U scikit-learn```. The first few blocks in the notebook are setting up for the rest of our code base. We start with a few imports and then loading our datasets. We then manipulate our datasets to fit the tokenization methods we are using later on. The manipulation of our dataset occurs from the notebook block 3 until block 7. Here we are doing the following to preprocess our dataset: remove variant questions and changing the answer column to either a 1 or 0 (1 for correct, 0 for incorrect). This leaves us with a dataset that has 4 features: A question, choice a, choice b, and a label that is our correct answer. We then tokenize our data so we can set up for training. This set up ensures that new users do not have to adjust much if anything at all to preprocess our datasets. Just click run and go.

#### Part 2: Setting Up for Training

While running the blocks sequentially, we arrive to the sections with ```AutoModelForMultipleChoice``` and ```DataCollatorForMultipleChoice```. These sections are pivotal in configuring our models to align with our customized datasets, which is in the multiple-choice format. Following this configuration, we establish a dedicated function designed to compute our evaluation metrics. In our case, we employ the F1 score as the primary metric to assess the performance of our models. This metric provides valuable insights into the models' precision and recall, offering a comprehensive understanding of their effectiveness in tackling the given task.

#### Part 3: Training

In this phase, we start the pivotal stage of model training. We first establish the model's architecture and configure the parameters. Additionally, we prepare the datasets required for both training and evaluation purposes. Following the setup, the tokenizer is used to facilitate the conversion of raw textual data into a format digestible by the model. Subsequently, the data collector orchestrates the organization of data, ensuring seamless processing throughout the training and evaluation phases.

To discover the proper functionality of our trainer, a test block is executed to validate its performance. Once verified, we proceed to evaluate the model's effectiveness on the designated evaluation dataset. The results are then computed and presented in a human-readable format, providing valuable insights into the model's accuracy and performance. To streamline this process and enhance reproducibility, we encapsulate the entire training and evaluation pipeline within the 'trainAndEval' function, offering a cohesive and efficient solution for model training and assessment.

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

The outputs are the answer choice (1 or 0 from our earlier preprocessing) that are then evaluated for accuracy using F1 score. This is hardcoded so please do not forget to change these when you run your versions. These evaluation results are then stored in lists to more easily graph the results. To provide a comprehensive analysis, the results are visually represented through bar charts categorizing the models into Baseline, Sentiment Analysis, and SQUaD types. Additionally, the last bar chart showcases the top-performing models from each category, offering a clear comparison of their respective performances.

See [here](https://docs.google.com/presentation/d/1K4x0OJyhAfyJciX1ozsdWsNbCaIEuzyqk6jUJZNDq2g/edit?usp=sharing) for our in class materials discussing our procedure and results.

### Final Results Expected Performance

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
