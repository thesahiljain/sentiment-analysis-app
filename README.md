# Sentiment Analysis API using transformers

## Introduction
<p> In this project, I have developed a deep learning model using tensorflow and huggingface's transformers to predict sentiment of a given textual input. I have used the pretrained ALBERT model in the transformers library and fine tuned it on labelled dataset of tweets. The model has been deployed on a docker container as a REST API using Flask. The user can send a POST request to the model with text input in JSON format. The API will return the sentiment value of the text (Positive/Negative) and it's polarity score.</p>

## Getting Started
<p>In order to run this model, you will need Docker and Git installed in a Linux environment.</p>

### Cloning the repository
<p>Run the following commands to clone the repository and enter the root directory of our Docker code</p>

> git clone https://<span></span>github.com<span></span>/thesahiljain/sentiment-analysis-app <br>
> cd Docker

### Building the docker container
<p>First, you need to build the docker container. To do so, run the following command</p>

> sudo docker build --tag sentiment-analysis-app .
<p>Here, I have tagged my docker container as "sentiment-analysis-app". You can use any other tag name of your choice. Also, several dependencies like tensorflow, transformers and flask will be installed in the image during this phase, so make sure you have enough data</p>

### Running the docker container
<p>Now we could run our docker container by using the following command</p>

> sudo docker run -p 8080:5001 sentiment-analysis-app

<p> This will start our flask server locally. The API run as port 5001 within the container, which you can modify in the Dockerfile and app.py file. The API runs at port 8080 in the host PC. You can modify that by just changing the value in the above command.</p>

## Infering the sentiment of the text
<p> Now, we can use our API to infer sentiment value of text. Here is an example using curl command </p>

> curl -H "Content-Type: application/json" -X POST -d '{"text":"This is not a bad day at all!"}' http://<span></span>0.0.0.0:8080/

<p> The response shall be a JSON value with 3 fields. The "success" field tells if the inference was successfully made or not. The "sentiment" field tells the binary value of the sentiment inferred. The "polarity_score" is the probability value of our inference, 0 being the most negative and 1 being the most positive</p>

> {"polarity_score":0.9784,"sentiment":"Positive","success":true}
