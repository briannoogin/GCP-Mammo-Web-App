# GCP-Mammo-Web-App

Breast cancer diagnosis using a CNN. Used CBIS-DDSM breast cancer dataset. Model is developed, trained, and hosted with Google Cloud Platform.

### Prerequisites

What things you need to install the software and how to install them

```
Google Cloud 
Keras and the pre-reqs
Tensorflow and the pre-reqs
Numpy
NBIA Data Retriever (For downloading files from official database)
```
## Getting Started

Sign up for an GCP account
Login in GCP with gcloud init to setup environment
Uncomment the first line in gcloud.run.sh or local_gcloud_run.sh to save login credentials locally

## Data Details:
Three classes: Benign, Benign without callback, Malignant 

Class Distrubution:
Train: Benign:1105, Benign without callback:578, Malignant:1181, Total: 2864 
Test: Benign:324, Benign without callback:104, Malignant:276, Total: 704

## Deployment

Configure config.yaml for specific Google ML Engine instance
Run gcloud.run.sh

## Built With

* [Google Cloud Platform](https://cloud.google.com/) - The cloud platform used
* [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/tensorflow/technical-overview) - Service used to train and test the CNN model
* [Google Cloud App Engine](https://cloud.google.com/appengine/) - Service used to host the REST API (future plans)

## Authors

* **Brian Nguyen** - *Initial work* - [briannoogin](https://github.com/briannoogin)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


