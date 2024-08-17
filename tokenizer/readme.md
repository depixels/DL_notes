# expand llama tokenizer (add chinese words)
# Jupyter version
## 1.prepare data
Only for practice, so just use a small corpus, you can use your own corpus in the same way. This step mainly extracts  useful information of json file and turns json file into txt file.
+ [CLUECorpusSmall-new2016zh (google drive)](https://drive.google.com/file/d/1TMKu1FpTr6kcjWXWlQHX7YJsMfhhcVKp/view?usp=drive_link)
+ [CLUECorpusSmall-baike2018qa (google drive)](https://drive.google.com/file/d/1_vgGQZpfSxN_Ng9iTAvE7hM3Z7NVwXP2/view?usp=drive_link)
## 2.make corpus
merge the preprocessed txt files into one txt file, which is the corpus used to train tokenizer.
## 3.train tokenizer
use sentencepiece to train tokenizer with corpus.hard to train using CPU and the corpus is too large.
## 4.test tokenizer
test the tokenizer with a sentence.

the whole code is run in jupyter notebook. [colab link](https://colab.research.google.com/drive/1zqaLBWpkgLvWBCEuhTJfKEwYF-JDxcmM?usp=drive_link)


