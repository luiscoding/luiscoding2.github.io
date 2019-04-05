---
layout:     post
title:      Table Understanding Paper
subtitle:    
date:       2018-09-10
author:     Lu Zhang
header-img: img/images/img-1904030.jpg
catalog: true
tags:
    - Paper
    - Table Understanding
---

## [Understanding Table on the Web](https://homes.cs.washington.edu/~jwang/publications/webtables.pdf)
### Problem Model: 
Define understanding: associating the table with one or more semantic concepts in a general purpose knowledge base  associate table to semantic meaning 
Probase: general-purpose taxonomy: conceptualize the information in web tables
Process: header detector->header generator->entity detector->if the quality is too low, discard the table 

### Method:
Knowledge Taxonomy:
Not hearst patterns
linguistic pattern:what is the A of I?
Evaluate: Plausibility Ambiguity

Header Detector:
Probability 

Header Generator:
Use top-k candidate concepts to generate

Entity Detector:
Entity column contain entities of the same concept
Header contain attributes which describe entities

### How to use:
1.Used to deal with the weather data table 
2.Generate the meta-data of certain tables based on limited information of the tables.


## [Finding Related Tables](http://i.stanford.edu/~anishds/publications/sigmod12/modi255i-dassarma.pdf)

### Problem is challenging for two main reasons:
1.The schemas of the tables in the repository are partial at best. schema needed for reasoning about the relatedness 
2.Different ways and degrees to which data can be related

### Problem Model:
A corpus of tables: T,(with partial meta-data), a table T1
Goal: return a list of tables in T related to T1

### Framework defining the relatedness:
1. T1 T2->T T1= Q1(T) T2= Q2(T)
2. T coherent 
3. Q1 Q2 similar structure

Entity Complement
Schema Complement

Methods:
1. EC 
relatedness between a pair of entities, entities sets

2.Schema Similarity
schema auto-complete score, the higher the likelihood of seeing these new attributes, the highter is the score

Experiment:
Metric:
For each query table, generate top-k related tables
Augmenting table search:
related queries , related table
related tables can be used as an important feature in tuning keyword search results

Scale Up:
General filtering-based approach: hash function

### How to use: 
1. Explore a more complex combination of relatedness of entity compliment and schema compliment 
2. Base on the related tables and the partial meta-data  in paper 1, then we can use similar tables to understand a new table.

## [TabEL: Entity Linking in Web Tables](http://iswc2015.semanticweb.org/sites/iswc2015.semanticweb.org/files/93660385.pdf)

### Main Idea: 
Semantic interpretation of tables, i.e, the task of converting web tables into machine -understandable knowledge 

### Problem Model : 
Entity Linking Web Pages, Given a table T and KB k , identify and link potential mention in cells of T to its referent entity e in k. 

### Existing Method: 
employ graphical models to jointly model three types: Entity Linking, column type identification, and relation extraction Strong Assumption: column types and relations can be mapped to pre-defined, which makes the infomation incomplete or noisy strict mapping 

Method: New alternative mapping KB , incorporate type and relation through a graphical model of soft constraints.

Step:
1. Find potential mention 
2. Candidate Generation 
3. Disambiguation 
Entities in a row/column tend to be related.
4.M(LR) rank based on features

a.Prior Probability features , estimated from hyperlinks on the Web

b.Semanticc relatedness features, measure the coherance between entities

c.Mention-Entity similarity deatures 

d.Existing Link 

e.Surface Features

Evaluation Metric:
macro-averaged precision, recall and F1 metrics


## [Methods for Exploring and Mining Tables on Wikipedia](http://poloclub.gatech.edu/idea2013/papers/p19-bhagavatula.pdf)

###Main Idea: 
Search and mining a knowledge base extracted from Wikipedia data tables. Make a relevant addition, correlation mining 
A prototype information exploration system: WikiTables
Focus wikipedia, relevant join on relevant data, rather than on tables

extracted table: Restricted tables belong to the HTML class extraction quality: precision, and recall 

finding relevant joins: A triplet contains a source column, matched column and a candidate column,

correlation mining: calculate pearson correlation between the numeric columns 

table search: a trained linear ranking model with coordinate ascent

### Experiments:
In the experiments,  because this system is different from previous work, so for the following task, they put up  corresponded evaluation metrics
1. relevant join 
previous work mainly use schema complement as an indicator of relatedness of tables and using this relatedness to improve table search 

2. correlation mining 
evaluated on the task using 100 randomly sampled pairs of numeric columns which are manually classified 

3. table search
evaluated based on the number of queries for which a given method returned a result in top 5 that are relevant

### How to use:
After extraction of nomolized wikitable, use the system to search and join related columns.
Problem: The  style of tables used in this system is strict. for the Datamart, we need to modify the model to deal with varied tables

## [Recovering Semantics of Tables on the Web](http://www.vldb.org/pvldb/vol4/p528-venetis.pdf)
### Main Idea: 
Enrich the table with additional annotations to recover the semantics of tables. 
i.e Some headers don't exist add Annotations 
Problem Model: 
Tables: no name, may not have names for attributes, values in a cell is single type 
Goal: Add annotations to explicitly expose the semantics of tables. 

Two kinds of queries: 
a. Find a property of a set of instances or entities 
b. Property of an individual instance

Method: 
2 databases: isA(instance, class) and relations1, database(argument, predicate, argument2)

Experiments：
Evaluate the quality of the table annotations and their impact on table search

### How to use: 
With this method, first we label the column, and the use the pattern to add additional information to the data. 
