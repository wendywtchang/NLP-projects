# :runner: NLP Projects
> Side projects of NLP works.  

## Overview
* Natural Language Processing
  * [Named Entity Recognition](#named-entity-recognition)
  * [Dialog system](#dialog-system)
  * [Topic Modeling](#topic-modeling)
  * [Emotion analysis](#emotion-analysis)
  * [Machine Tanslation](#machine-translation)
* Speech Recognition
  * [Auromatic Speech Recognition](#auromatic-speech-recognition)
* Annotation / Web Design
  * [Annotation Samples](#annotation-samples)
* Working
  * Language Modeling
  * Coreference Resolution in Cooking Recipe
  * Cross-modal Coref. in Recipes


## :star: Natural Language Processing
### Named Entity Recognition
* #### Slot Filling Task
  * A natural language understanding task to fill slots in dialog system with a digital assistant command dataset. 
  * Using Pytorch and BERT model.

### Dialog system
* #### Relation Prediction of Semantic Parser in Knowledge Base Question Answering
  * Predict the relation given a question from Wikidata. 
  * Question sentences are processed with BERT embedding and predicted by a MLP model.

### Topic Modeling
* #### [Topic Modeling in COVID-19 Papers in PubMed](./topic_modeling/LDA_Covid19.ipynb)
  * Apply Latent Dirichlet Allocation (LDA) Model, an unsupervised approach. 
  * Explore different topics in COVID-19 related papers' abstract from [PubMed](https://pubmed.ncbi.nlm.nih.gov/).
  * Result: The best model categorized the articles into 15 topics with the (log) perplexity of -7.82 and the coherence score of 0.56. 
  * Visualization: pyLDAvis library.

### Emotion analysis
* #### [Sentiment Classifiers](./emotion_analysis/Sentiment_Classifier.ipynb) 
  * Dataset: Multimodal EmotionLines Dataset - [MELD](https://affective-meld.github.io/) 
  * Building a LSTM model by using keras to classify 7 different sentiments. 
  * Tensorfolw / Keras
  * Ekman 6 emotions + Neutral.
  * Results: 30 epoch, F1 = 26%
  * Visualization: Matplotlib
* #### Extract sentiment stimulus
  * Perform a name entity recognition task to extract sentences with emotional information.

### Machine Translation
* #### [Seq2Seq MT (Results)](./neural_machine_translation/NMT_seq2seq.ipynb), [Main Python Script](./neural_machine_translation/seq2seq_model_v1.7.py)
  * Dataset: [Multi30k](https://aclanthology.org/W16-3210.pdf): Multi30K - Multilingual English-German Image Descriptions
  * Task: Translate German to English
  * Model: Sequence to sequence model, two LSTM as encoder and decoder
  * Results: Bleu score 18.04 (50 epoch)


## :musical_note: Speech Recogniton
### Automatic Speech Recognition
* #### [ASR in Mandarin-English Code-Switching Dataset](./asr/)
  * Building an ASR system from scratch
  * Dataset: [SEAME](https://www.semanticscholar.org/paper/SEAME%3A-a-Mandarin-English-code-switching-speech-in-Lyu-Tan/f28cb37e0f1a225f0d4f27f43ef4e05eee8b321c), 30 hr Mandarin-English codeswitching dataset
  * Hidden Markov Model (acostic model) + N-gram Languae Model  
  * Result: Word Error Rate (WER): 64.53%. Character Error Rate (CER): 54.06%


## :pencil2: Annotation
### Annotation Samples
* #### The design layout of the annotation on Amazon Mechanical Turk.
  * Includeing two annotation design pages: 
  * Coreference (text) annotation 
  * Bounding Box (image) annotation 
  * Using HTML, CSS, Javascript and Amazon Crowd HTML Elements.

## :blush: Contact
Feel free to contact me! [@wendywtchang](<mailto:wentseng.chang@gmail.com>) 
