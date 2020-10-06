# Relation prediction model in dialog system:   
> This is a project of building a relation predictor in semantic parser of an knowledge base question answering system.  
  
> Dataset: KBQA dataset of Wikidata build by [@csqa](https://amritasaha1812.github.io/CSQA/).  
> Goal: To predict the relation connection between subject & object entities.  
> Integration: Adviser dialog system framework [@adviser](digitalphonetics.github.io/adviser/).  
  
> This is a collaboration work with [@e-tony](https://github.com/e-tony).

## Overview
* [Model](#model)
* [Integration](#integration)
* [Toolkits](#toolkits)
* [Setup](#setup)
* [Demo](#demo)

## Model   
* MLP: Multilayer perceptron, a simple and efficient neural network model.  
* Bert embeddings: Contextualized word embeddings.  

## Integration  
* Adviser dialog system: 
* Open-source multi-modal dialog system. 
* Use Publish/Subscribe pattern.
* Allows developers to test their own design with sub-modules and to flexibly integrate them into the system.

## Toolkits
* pytorch 
* tensorflow

## Setup
1. Install adviser toolkits & environment   
2. Add `dialog_system` file under the directory: `adviser/adviser/examples/qa/` and change the file name into `semanticparser`
3. Run `run_chat_mlp.py` under `adviser/adviser/`

## Usage
1. Enter `python run_chat_mlp.py qa`  
2. The system should say `Welcome to adviser world knowledge QA system, ...`
3. Ask whatever you like
4. Enter `bye` to leave the system

## Demo
Here is a demo of using adviser system in QA domain with our model.  
![Adviser QA Demo](./img/adviser_demo.png)  

* It can answer simple worldwide knowledge questions. (latest update: Oct. 2020)
* The model reaches macro F1 83%


