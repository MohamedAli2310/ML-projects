## Description
Given a dataset of arm movement patterns collected from 5 individuals while they walked in a standard corridor.
The data was collected in two different sessions.
The data collected for the same user in different sessions is supposed to be similar (known as intra-user similarity)
while the data collected from different users, regardless of the session, are supposed to be different known as
inter-user dissimilarity from each other.

The data is stored in the following folder structure: (More details are provided in Data_description.txt)
<Dataset>/<UserId>/<Training>
<Dataset>/<UserId>/<Testing>

Implemented:
(1) Preprocessor 
(2) Comparators 
(3) Person
(4) Evaluator
(5) ResultAnalysis

