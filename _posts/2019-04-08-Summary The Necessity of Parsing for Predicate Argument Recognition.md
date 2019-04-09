---
layout:     post
title:      Summary The Necessity of Parsing for Predicate Argument Recognition
subtitle:  
date:       2019-04-08
author:     Lu Zhang
header-img: img/images/img-190403.jpg
catalog: true
tags:
    - NLP
    - Paper
---

## Main Work:
For label semantic roles, there are statistical systems. In this paper,
1. Quantify the effect of parser accuracy on systems' performance
2. Examine whether a flatter "chunked" representation of the input can be as effective for semantic role identification 


### Semantic role 
Identify semantic role is a crutial part of interpreting text, and can serve as intermidiate step of machine translation.

### Data
Propbank corpus: Roles are defined on a per-predicate basis. Core arguments are simply numbered, remaining are given labels as temporary or locative. 
Only verbs
FrameNet:Semantic frames, Nouns, adjectives. 

### Features
Semantic role is calculated from the following features: 
Phrase Type: Noun Phrase, Verb Phrase
Parse Tree Path: Syntactic relation of a constituent to the predicate 
Position: Indicated whether constituent to be labeled occurs before or after the predicate defining the semantic frame.
Voice:  Active, Passive 
Head Word: Lexical Dependencies 

Question: 
Why path feature is useful for unknown boundary condition?

### Experiment 
In order to see how much of the performance degradation is caused by the difficulty of finding exact argument boundaries in the chunked representation, we can relax the scoring criteria to count as correct all cases where the system correctly identies the rest chunk belonging to an argument.


### Semantics 
Semantics is a big topic in NLP field, semantics, in general, is the study of meaning. What do expressions mean? How can we extract meaning from representations used in language? Why is semantic is important? 

**Lexical Semantics**  meanings of words
**Compositional Semantics** how meanings are put together 
**Moel- Theoritic Semantics** Map human language expressions to a logical representation taht is instantiated in a model 
Scope Ambiguous: 
When we talk about semantics, we can look up the dictionaries to check the meaning of the words. 
Dictionaries: 
WordNet- w3 ontology 
Probank  
FrameNet 
[OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19), word sense disambiguation. 






