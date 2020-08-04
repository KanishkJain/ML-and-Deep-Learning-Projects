# ML-and-Deep-Learning-Projects

My name is Kanishk Jain. I'm an alumni from IIT-Bombay, currently working in P&G. I'm extremely passionate about AI and ML, and enjoy reading and applying cutting edge innovations and solutions across multiple domains. I've experience in using both R and Python for ML and Deep Learning, primarily focussing on Structured Datasets. Below is a repository of major blogs and projects that I've worked upon in Public Domain. Professionaly, I've experience in using ML in Industry 4.0 settings for - Predictive Maintenance, Process Optimisations and Image Processing in plants.

****Blogs**** -
1. **Evolution of NLP** - A series of articles showing improvements in the algorithms used of various Natural Language Processing Tasks. Over last few years, NLP has seen some of the most ground-breaking innovations, that have transformed the way we used to think about NLP tasks. But, it is difficult to keep pace with these innovations, and all the resources present online are either not up to date, or too fragmented - no one place to learn it all. This inspired creation of these series, which starts from Basics - *Bag of Words*, *N-grams*, *TF-IDF* and goes to the SOTA transformer models - *BERT, RoBERTa and XLNet*. Sharing the first two articles of this series
  * [Evolution of NLP - Part 1 - Bag of Words, TF-IDF](https://medium.com/@jainkanishk001/evolution-of-nlp-part-1-bag-of-words-tf-idf-9518cb59d2d1)
  * [Evolution of NLP - Part 2 - RNNs](https://medium.com/@jainkanishk001/evolution-of-nlp-part-2-recurrent-neural-networks-af483f708c3d)
  * [Evolution of NLP - Part 3 - Transfer Learning with ULMFit](https://medium.com/@jainkanishk001/evolution-of-nlp-part-3-transfer-learning-using-ulmfit-267d0a73421e)

Machine Learning for Structured Dataset -
1. [Analytics Vidya JanataHack - E-Commerce Analytics ML Hackathon](https://github.com/KanishkJain/Analytics-Vidhya-JantaHack-) - **2nd Rank** (Private LB) out of 2500+ participants. Used a 5-fold CV Light GBM Model, along with feature engineering to extract other-wise difficult to identify insights, to predict the gender of a shopper based on it's browsing history on an e-commerce platform; achieving 94.8% accuracy.
2. [Practical Machine Learning](https://www.coursera.org/account/accomplishments/records/QA7US5DYN7ZQ) - Course 8 of Data Science Specialization from John Hopkins University. Completed following projects as course assignments -
* *Activity Classifier* - Used Logistic Regression, LDA, Random Forest, Decision Trees, using the data from accelerometers across multiple places in body, to identify whether a certain activity is being done correctly or not.

Convolutional Models -
1. [Analytics Vidya JantaHack Computer Vision Hackathon](https://github.com/KanishkJain/ML-and-Deep-Learning-Projects/tree/master/Convolution%20NN%20Models) - **6th Rank** (Private LB) out of 10000+ participants. Used an ensemble of Pre-Trained models - EfficientNet, InceptionResnetV2, InceptionNetV3 and VGG19 using fast.ai library to predict whether the vehicle in image is "Emergency" (Fire-Trucks, Ambulances, Police) or "Non-emergency"
2. [Semantic Segmentation using FCN - Course Project](https://github.com/KanishkJain/Semantic-Segmentation-using-FCNs) - Used Fully Convolution Networks, essentially stripping off all the pooling layers and leaving only Convolutions in VGG-16, and generate predictions for individual pixels within an image and classify it into the right category. Dataset used for this was the same as one used in Pascal VOC 2012 competition.
3. [Tensorflow in Practice - Convolution Neural Networks in Tensorflow](https://www.coursera.org/account/accomplishments/records/UDTUDWKQVB54) - Course 1 of the Tensorflow in Practice Specialization. Completed following projects as course assignments -
 * *Sign Language Classifier* - Used Transfer Learning, loading pre-trained parameter for InceptionNet to look at images of signs and predict the characters they mean in Sign Language.
4. [deeplearning.ai - Convolutional Neural Networks](https://www.coursera.org/account/accomplishments/records/487KFZGDULS6) - Course 4 of Deep Learning Certification. Completed following projects as course assignments -
 * *Car Detection with YOLO* - Used YOLOV3 (You Look Only Once) algorigthm to generate bounding boxes around different objects in an image
 * *Art Generation with Neural Style Transfer* - Used Pre-Trained VGG19 over two images - content and style - transferring the style of the second image to the first one by optimizing for the pixel-level overall loss fuction
 * *Face Detection* - Used Pre-Trained InceptionV3 model, along with Triplet Loss function optimization to build a face verification tool


Sequence Models - 
1. [Analytics Vidya JanthaHack NLP Hackathon](https://github.com/KanishkJain/ML-and-Deep-Learning-Projects/blob/master/Sequence%20Models/Steam%20Reviews%20Classifier%20with%20BERT.ipynb) - **Top-50** (Private LB) out of 2500+ participants. Used pre-trained BERT (Bi-Directional Encoder Representation from Transformer) in Tensorflow as base and added classification head on top to predict sentiment of user reviers for Steam Games, achieving 90%+ accuracy.
2. [Analytics Vidya Jantahack Recommender Systems Hackathon](https://github.com/KanishkJain/ML-and-Deep-Learning-Projects/blob/master/Sequence%20Models/Recommender%20System%20-%20FastAI%20Language%20Model%20(1).ipynb) - **Top-25** (Private LB) out of 12000+ participants. Used AWD-LSTM, pre-trained on Wikipedia Corpus, using fast.ai library to recommend next 3 challenges user is most likely to take in an online learning platform, based on first 10 challenges for each user, achieving 0.23 Mean Average Precision (MAP)
3. [Twitter Sentiment Analysis](https://github.com/KanishkJain/ML-and-Deep-Learning-Projects/blob/master/Sequence%20Models/Twitter%20Sentiment%20Analysis.ipynb) - Used Twitter API along with vaderSentiment Library to study the sentiment of recent tweets on a selected topic and word cloud showing the most shared tokens/words in tweets in poistive and negative sentiment tweets.
4. [deeplearning.ai - Sequence Models](https://www.coursera.org/account/accomplishments/records/W479BMQL3WL3) - Course 5 of the Deep Learning Certification. Completed following projects as course assignments - 
  * *Trigger Word Detection* - Used Convolution Layer, along with multiple layers of GRU to create a triggered word detector similar to Amazon's Alexa, or Apple's Siri, which gets activated when right keyword is used during speech.
  * *Neural Machine Translation* - Used Sequence to Sequence models based on Attention Mechanism, to translate human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25")
5. [Tensorflow in Practice - Natural Language Processing in Tensorflow](https://www.coursera.org/account/accomplishments/records/CFT97H5YU8H9) - Course 3 of the Tensorflow in Practice Specialization. Completed following projects as course assignments -
  * *Sarcasm Detection from News Headline* - Used Convolution Layer, along with multiple Bidirectional LSTM layers in Keras to predict whether a headline is sarcastic or not based on it's headline
  * *IMDB Movie Review Sentiment Analysis* - Used Pre-Trained Glove Embeddings, along with LSTM layers to predict the sentiment (good, neutral or bad) for IMDB Movie Review dataset.
  
Online Courses/Specialization -
1. [**Deep Learning Specialization - deeplearnig.ai - Coursera**](https://www.coursera.org/account/accomplishments/specialization/HLQBKARHV996)
2. [**Tensorflow in Practice - deeplearning.ai - Coursera**](https://www.coursera.org/account/accomplishments/specialization/NM98KUR9DDTV)
3. [**Machine Learning - Stanford University (Andrew Ng) - Coursera**](https://www.coursera.org/account/accomplishments/records/5XM3MSJ46P5B)
4. [**Practical Machine Learning - John Hopkins University - Coursera**](https://www.coursera.org/account/accomplishments/records/QA7US5DYN7ZQ)
5. Mutliple courses from **Datacamp**, across Data Wrangling, Data Visualization, Machine Learning and Statistics
