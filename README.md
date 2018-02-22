# HeavyWater Machine Learning Problem

### Purpose

The purpose of this problem is to evaluate your abilities in several dimensions at once.

  1. Do you understand the principles of ML/AI/data science/<insert fancy other term here>
  1. Can you build something that works
  1. Do you have a grasp of the tool chain from code on your local to code in production
  1. Can you explain your design and thinking process
  1. Are you excited by learning and challenges


### Problem Statement

We process documents related to mortgages, aka everything that happens to originate a mortgage that you don't see as a borrower. Often times the only access to a document we have is a scan of a fax of a print out of the document. Our system is able to read and comprehend that document, turning a PDF into structured business content that our customers can act on.

This dataset represents the output of the OCR stage of our data pipeline. Since these documents are sensitive financial documents we have not provided you with the raw text that was extracted. Instead we have had to obscure the data. Each word in the source is mapped to one unique value in the output. If the word appears in multiple documents then that value will appear multiple times. The word order for the dataset comes directly from our OCR layer, so it should be _roughly_ in order.

Here is a sample line:

```
CANCELLATION NOTICE,641356219cbc f95d0bea231b ... [lots more words] ... 52102c70348d b32153b8b30c
```

The first field is the document label. Everything after the comma is a space delimited set of word values.


### Your Mission

Should you choose to accept it ...

Train a document classification model. Deploy your model to a public cloud platform (AWS/Google/Azure/Heroku) as a webservice, send us an email with the URL to you github repo and the URL of you model. Also, we use AWS so we are partial to you using that ... just saying.


### Measurement Criteria

We will measure your solution on the following criteria, if we like what you have produced we will ask you for a code review to discuss your thinking:

  1. Does your webservice work?
  1. Is your hosted model as accurate as ours? Better? (think confusion matrix)
  1. Your code, is it understandable, readable and/or deployable?
  1. Do you use industry best practices in training/testing/deploying?
  1. Do you use modern packages/tools in your code and deployment?


### A few more details

Webservice spec:

- RESTful API
- Respect content-type header (application/json and text/html minimum other bonus)
- Discoverable from root path
- URL encoded GET parameter "words" returns predicted document type (confidence is a bonus) in field "prediction" and "confidence"
- HTML pages should be readable by a human and allow for action, aka input field and submit buttons etc.
