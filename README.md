# ML-and-Deep-Learning-Projects
My Public repository of Machine Learning and Deep Learning Projects

Blogs -
1. *Evolution of NLP* - A series of articles showing improvements in the algorithms used of various Natural Language Processing Tasks. Over last few years, NLP has seen some of the most ground-breaking innovations, that have transformed the way we used to think about NLP tasks. But, it is difficult to keep pace with these innovations, and all the resources present online are either not up to date, or too fragmented - no one place to learn it all. This inspired creation of these series, which starts from Basics - *Bag of Words*, *N-grams*, *TF-IDF* and goes to the SOTA transformer models - *BERT, RoBERTa and XLNet*. Sharing the first two articles of this series
  a. [Evolution of NLP - Part 1 - Bag of Words, TF-IDF](https://medium.com/@jainkanishk001/evolution-of-nlp-part-1-bag-of-words-tf-idf-9518cb59d2d1)
  b. [Evolution of NLP - Part 2 - RNNs](https://medium.com/@jainkanishk001/evolution-of-nlp-part-2-recurrent-neural-networks-af483f708c3d)

Sequence Models - 
1. [Analytics Vidya JanthaHack NLP Hackathon](https://github.com/KanishkJain/ML-and-Deep-Learning-Projects/blob/master/Sequence%20Models/Steam%20Reviews%20Classifier%20with%20BERT.ipynb) - 42nd (Private LB) out of 2500+ participants. Used pre-trained BERT (Bi-Directional Encoder Representation from Transformer) in Tensorflow as base and added classification head on top to predict sentiment of user reviers for Steam Games, achieving 90%+ accuracy.
2. [Analytics Vidya Jantahack Recommender Systems Hackathon](https://github.com/KanishkJain/ML-and-Deep-Learning-Projects/blob/master/Sequence%20Models/Recommender%20System%20-%20FastAI%20Language%20Model%20(1).ipynb) - 25th (Private LB) out of 12000+ participants. Used AWD-LSTM, pre-trained on Wikipedia Corpus, using fast.ai library to recommend next 3 challenges user is most likely to take in an online learning platform, based on first 10 challenges for each user, achieving 0.23 Mean Average Precision (MAP)
3. [deeplearning.ai - Sequence Models](https://www.coursera.org/account/accomplishments/records/W479BMQL3WL3) - Course 5 of the Deep Learning Certification. Completed following projects as course assignments - 
  a. *Trigger Word Detection* - Used Convolution Layer, along with multiple layers of GRU to create a triggered word detector similar to Amazon's Alexa, or Apple's Siri, which gets activated when right keyword is used during speech.
  b. *Neural Machine Translation* - Used Sequence to Sequence models based on Attention Mechanism, to translate human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25")
4. [Tensorflow in Practice - Natural Language Processing in Tensorflow](https://www.coursera.org/account/accomplishments/records/CFT97H5YU8H9) - Course 2 of the Tensorflow in Practice Specialization. Completed following projects as course assignments -
  a. *Sarcasm Detection from News Headline* - Used Convolution Layer, along with multiple Bidirectional LSTM layers in Keras to predict whether a headline is sarcastic or not based on it's headline
  b. *IMDB Movie Review Sentiment Analysis* - Used Pre-Trained Glove Embeddings, along with LSTM layers to predict the sentiment (good, neutral or bad) for IMDB Movie Review dataset. 
