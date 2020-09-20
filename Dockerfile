FROM tensorflow/tensorflow:latest

LABEL version="1.0.0" by="clement.bolin@epitech.eu"

WORKDIR /deepLearningGaz 


# Install python
RUN apt update -y && apt-get install python3-dev python3-pip -y
# RUN apt-get install git -y
RUN pip3 install --upgrade pip

# Install dependencies
RUN pip3 install Theano
RUN pip3 install keras
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install -U scikit-learn

COPY . .
EXPOSE 8080

CMD [ "python3", "main.py"]
