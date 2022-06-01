# Semantic-Based-K-Means

## Table of content
- [1. Paper](#1-Paper)
  * [1.1 Abstract](#11-abstract)
  * [1.2 Introduction](#12-introduction)
- [2 Code](#2-code)
  * [2.1 reading_input.py](#21-reading_inputpy)
---

### 1.Paper
Semantic-based K-means clustering for microblogs exploiting folksonomy. Journal of Information Processing Systems, 14(6), 1438-1444.
http://xml.jips-k.org/full-text/view?doi=10.3745/JIPS.04.0097

#### 1.1 Abstract
Recently, with the development of Internet technologies and propagation of smart devices, use of microblogs
such as Facebook, Twitter, and Instagram has been rapidly increasing. Many users check for new information
on microblogs because the content on their timelines is continually updating. Therefore, clustering algorithms
are necessary to arrange the content of microblogs by grouping them for a user who wants to get the newest
information. However, microblogs have word limits, and it has there is not enough information to analyze for
content clustering. In this paper, we propose a semantic-based K-means clustering algorithm that not only
measures the similarity between the data represented as a vector space model, but also measures the semantic
similarity between the data by exploiting the TagCluster for clustering. Through the experimental results on
the RepLab2013 Twitter dataset, we show the effectiveness of the semantic-based K-means clustering
algorithm. 

#### 1.2 Introduction
Recently, with the development of Internet technologies and propagation of smart devices, users are
able to easily access various information. In the Web 2.0 environment, users can create and share
various media such as images and videos through a social media service, and the quantity of those
content has been increasing daily. In particular, users have been exposed to a flood of information
because of the rapid growth of microblog services such as Twitter, Facebook, Instagram, and Tumblr.
Twitter is one of the most famous microblogs, where people express their thoughts or opinions using a
short message with the limitation of 140 characters, referred to as a tweet. The user can freely express
their thoughts or feelings about any topic by creating a tweet and constructing their own social network.
Twitter is used by various celebrities, such as singers, athletes, actors, and politicians, as well as ordinary
people. According to research, on December 31, 2013, there were already more than 240 million active
users per month, spanning nearly every country in the world. Approximately 500 million tweets are
created every day, and the number of created tweets is continually increasing.
It is not easy to find meaningful information from a large group of tweets after users are offline for a while because newly posted content is continuously generated on their timelines. Therefore, microblog
users have to check all of the newly posted content to find the most interesting information. To address
this problem, many cluster algorithm techniques have been proposed for arranging microblog contents.
These technologies have become especially important with the advent of smart devices and wearable
devices. However, microblog contents generate less data than general documents for clustering because
the limited information. Therefore, it is essential to overcome this by exploiting an external knowledge
base from the collective intelligence of folksonomy when analyzing and detecting microblogs.
In this paper, we propose a semantic-based K-means clustering algorithm that not only measures the
similarity between the tweets represented by a vector space model, but also measures the semantic
similarity between the tweets using TagCluster of Flickr for clustering a large number of tweets. The rest
of this paper is organized as follows: Section 2 introduces related work on clustering techniques for
tweets. Section 3 presents our proposed system. Experimental results are presented in Section 4. Finally,
Section 5 concludes the paper by discussing the direction for future work.

---

rest of the paper is available online on http://xml.jips-k.org/full-text/view?doi=10.3745/JIPS.04.0097

---

## 2. Code
In this section  a short description about  each file and some lines of codes  is available
### 2.1 reading_input.py
The information needed to crawl the tweets is provided by replab2013; However, due to respect to the privacy of users the text of tweets is not available thus you have to read them by your own. The information about each tweet are in some files which are named after some entity, for more information visit replab2013 <a href="http://nlp.uned.es/replab2013/">webpage</a>.Each file where saved as **.dat** file which were split into two part **1-train** **2-test** which were stored in such dir. -/training/tweet_info .

So Long story short, in this file we try to read all information from each file in order, then we save train and test file and even a file which contain both.

