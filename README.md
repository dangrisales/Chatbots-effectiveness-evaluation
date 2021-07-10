# Chatbots-effectiveness-evaluation
## Abstract 
Chatbots enable the automation of several components in customer service and allow the support of multiple users. Despite their multiple advantages, due to the large amount of conversations generated by a chatbot, it is difficult to  determine whether customer requests are well-addressed.  For practical reasons, chatbot's effectiveness is evaluated manually based upon a small sample (randomly chosen) of  conversations or through self-reported user satisfaction. This procedure does not guarantee the correct evaluation of the service because the sample is generally not large enough and self-reports might be influenced by different external factors not directly associated to the chatbot's functioning. This study proposes a methodology for automatic evaluation of chatbot effectiveness in real production environments. The analysis considers convolutional neural networks adapted for natural language processing, using two parallel convolutional layers to evaluate questions and answers independently. The proposed model also incorporates filters to extract features with multiple temporal resolution. This methodology is tested upon real conversations of chatbots that provide service to two different companies. The results are compared to baseline models based on classical techniques with different pre-trained word embedding models. According to our results, the proposed approach provides accuracies between 78.95\% and 80.18\%, which outperforms the best result of the baseline model by 2.9\%.
## Important Note
Unfortunately it is not possible to share the dataset due to privacy reasons of Pratech company. Similarly the train models cannot be shared because, as you know, they encode information of the training dataset