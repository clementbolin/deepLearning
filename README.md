# Deep Learning Training

## Why

For some time now, we have been hearing everywhere the words artificial intelligence, machine learning, deep learning. So I decided to start creating deep learning AI.

The objective of this repo is to share with you with explanations the exercises that I perform during the different tutorials and exercises that I follow. 

## Exercice 1 : Prediction in bank

The purpose of this exercise is to determine if a client X in a bank will leave the bank within the next 6 months.
in the **data folder** you will find the file ```Churn_Modelling.csv``` being the dataSet to be used for this exercise

example data in ```Churn_Modelling.csv```

    | RowNumber | CustomerId | Surname |CreditScore |Geography | Gender | Age | Tenure | Balance NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
    | 1         | 15634602   | Hargrave| 619        | France   | Female | 42  | 2      | 0                     | 1         | 1              | 101348.88       |   1    |

**CreditStore**: a client's ability to repay
**Tenure**: number of year since character is client in banck
**Balance NumOfProduct**: number of product in bank (example hash card)
**EstimatedSalary**: salary each year
**Exited**: 1 if client leave bank, 0 stay in bank

#### Which algorithm did I use

for my hidden layer I use Redresseur activate function, and for output layer I choice the sigmoid function.

### How to run exercice 1

#### Docker

[install docker for Linux](https://docs.docker.com/engine/install/ubuntu/)

[install docker for Mac](https://docs.docker.com/docker-for-mac/install/)

[install docker for windows](https://docs.docker.com/docker-for-windows/install/)

    git clone https://github.com/ClementBolin/deepLearning
    cd deepLearning
    docker build -t deeplearning .
    docker run -t deeplearning 

#### Without docker

first step install python3 and virtualenv

    git clone https://github.com/ClementBolin/deepLearning
    cd deepLearning
    pip install -r requirements.txt
    python3 main.py

## How to contribute

As I said the goal of this repo is knowledge sharing, so to contribute nothing simpler, just create a folder with the name of the exercise.
The folder should contain the following elements:

- README.md : who detail the goal of your exercice, how to run exercice
- SRC : all sources files, with comment how you explain each steps
