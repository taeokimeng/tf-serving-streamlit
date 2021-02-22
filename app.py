import streamlit as st
import matplotlib.pyplot as plt
import requests
import base64
import json
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

# server URL
# url = 'http://localhost:8501/v1/models/img_classifier:predict'
# url = 'http://localhost:8501/v1/models/img_classifier/versions/1:predict'

@st.cache
def load_mnist_data():
    # load MNIST dataset
    (_, _), (x_test, y_test) = load_data()
    # reshape data to have a single channel
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    # normalize pixel values
    x_test = x_test.astype('float32') / 255.0

    return x_test, y_test

def show(idx, title, x_test):
    fig = plt.figure()
    plt.imshow(x_test[idx].reshape(28, 28))
    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    # plt.show()
    st.pyplot(fig)

def data_plot(idx, x_test):
    fig = plt.figure(figsize=(0.6, 0.6))
    plt.imshow(x_test[idx].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.axis('off')
    st.pyplot(fig)

def data_plots(start, end, x_test):
    count = end - start + 1
    if count % 5 == 0:
        nrows = count // 5
    else:
        nrows = (count // 5) + 1

    fig, axes = plt.subplots(nrows, 5)
    for i, ax in enumerate(axes.flat):
        if i > count - 1:
            ax.set_axis_off()
            continue
        ax.imshow(x_test[start + i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        ax.axis('off')

    # fig = plt.figure(figsize=(3, 3))
    # for i in range(end - start + 1):
    #     plt.subplot(((end - start) // 5) + 2, 5, 1 + i)
    #     plt.imshow(x_test[start + i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    #     plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

def predictions_plots(start, end, x_test, y_test, predictions):
    count = end - start + 1
    if count % 5 == 0:
        nrows = count // 5
    else:
        nrows = (count // 5) + 1

    fig, axes = plt.subplots(nrows, 5)
    for i, ax in enumerate(axes.flat):
        if i > count - 1:
            ax.set_axis_off()
            continue
        ax.imshow(x_test[start + i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        if np.argmax(predictions[i]) != y_test[i]:
            ax.set_title(f"{np.argmax(predictions[i])}", color='r')
        else:
            ax.set_title(f"{np.argmax(predictions[i])}")
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

def make_prediction(instances, model, version):
    url = f'http://localhost:8501/v1/models/{model}/versions/{version}:predict'
    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions

def main():
    st.title("MNIST classifier")
    with st.spinner('Loading data...'):
        x_test, y_test = load_mnist_data()
    st.success('Data loaded')

    number = st.number_input('Insert the index', value=int(0), format='%d')
    if type(number) is int:
        data_plot(number, x_test)

    start, end = st.select_slider('Select the index range', options=list(range(30)), value=(1, 9))
    data_plots(start, end, x_test)

    model = st.radio("Select the model", ("MyLeNet", "LeNet"))
    version = st.radio("Select the version", (1, 2))
    if st.button('Request prediction'):
        predictions = make_prediction(x_test[start:end + 1], model, version)
        predictions_plots(start, end, x_test, y_test, predictions)
        # for i, pred in enumerate(predictions):
        #     st.write(f"Index {start + i} ... True Value: {y_test[i]}, Predicted Value: {np.argmax(pred)}")


if __name__ == '__main__':
    main()