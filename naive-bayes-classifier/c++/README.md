Since the dataset is large, it has been excluded from the repo. 

> *Dataset used*: http://ai.stanford.edu/~amaas/data/sentiment/

Extract the dataset maintaining the directory structure as in the `tree.txt` file.

Tranform the rows in `/dataset/train/labeledBow.feat` and `/dataset/test/labeledBow.feat` in the following format for the code to be able to parse lines as feature vectors.

- *train/labeledBow.feat*  

` 9 0:9 1:1 2:4 3:4 4:6 5:4 6:2 7:2 8:4 ... 24551:1 47304:1 @ `

- *test/labeledBow.feat* 

` @10 0:7 1:4 2:2 3:5 4:5 5:1 6:3 7:1 8:6 ... 15612:2 26903:1 # `