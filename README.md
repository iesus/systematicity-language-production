# Semantic Systematicity in Connectionist Language Production

Here is the code and trained specimens presented in the paper [Semantic Systematicity in Connectionist Language Production](https://www.mdpi.com/2078-2489/12/8/329/htm).

Abstract:
>Decades of studies trying to define the extent to which artificial neural networks can exhibit systematicity suggest that systematicity can be achieved by connectionist models but not by default. Here we present a novel connectionist model of sentence production that employs rich situation model representations originally proposed for modeling systematicity in comprehension. The high performance of our model demonstrates that such representations are also well suited to model language production. Furthermore, the model can produce multiple novel sentences for previously unseen situations, including in a different voice (actives vs. passive) and with words in new syntactic roles, thus demonstrating semantic and syntactic generalization and arguably systematicity. Our results provide yet further evidence that such connectionist approaches can achieve systematicity, in production as well as comprehension. We propose our positive results to be a consequence of the regularities of the microworld from which the semantic representations are derived, which provides a sufficient structure from which the neural network can interpret novel inputs.

You can cite this work/code with:
```
@Article{info12080329,
AUTHOR = {Calvillo, Jesús and Brouwer, Harm and Crocker, Matthew W.},
TITLE = {Semantic Systematicity in Connectionist Language Production},
JOURNAL = {Information},
VOLUME = {12},
YEAR = {2021},
NUMBER = {8},
ARTICLE-NUMBER = {329},
URL = {https://www.mdpi.com/2078-2489/12/8/329},
ISSN = {2078-2489},
DOI = {10.3390/info12080329}
}
```


The code uses theano and Python 2.7

[data](https://github.com/iesus/systematicity-sentence-production/tree/main/data) contains the corpus and the different train/test splits. There are some text files that can be used to get an idea of the sentences, but most files are in Pickle format, which is the format the code receives.

[outputs](https://github.com/iesus/systematicity-sentence-production/tree/main/outputs) contains the trained specimens and some outputs of them. Some values reported in the paper need to be rerun as they might not be here, which we will try to fix. However, all specimens in the paper are here, so any further tests or analyses are possible. Each trained neural network is relatively small that is why we can post all our specimens.

[production_main.py](https://github.com/iesus/systematicity-sentence-production/blob/main/production_main.py) is the main file. If you would like to reproduce our results, this is the one you should look at.

