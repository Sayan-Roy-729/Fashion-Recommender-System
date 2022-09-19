# 👗 **Fashion Recommender System** 👗

## 📃 **Project Details**
Build a fashion recommender system where if a user upload an image to the site, the model will search top 5 similar images from the image dataset and show them to the user. For high accuracy and success rate of Convolution Neural Networks (CNN) now a days, we have used `ResNet-50` as Transfer-learning model and used a technique called `Reverse Image Search` for this recommendation.

## 🌐 **How to run locally?**
1. As a Data Science/DL project, first have to get the dataset. We have used this [dataset from kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). Download to the loacl disk and extract the zip file. We only need the image files. The folder structure should like this:

```bash
- Fashion-Recommender-System
    |-- fashion-dataset
    |   |-- fashion-dataset
    |   |-- images
    |   |-- styles
    |   |-- images.csv
    |   |-- styles.csv
    |
    |-- And Other files & folders
```

2. Then we have used `python3.8`. You can download that from [here](https://www.python.org/downloads/release/python-3912) according to the OS and install to your machine. Then create a virtual environment named `venv` or anything like that. For more reference you can visit this [site](https://www.geeksforgeeks.org/python-virtual-environment).

```bash
pip install virtualenv
virtualenv venv
```
After that, activate the virtual environment.

3. Install the required libraries used for this projects. For that you can use below command. *(There are some large packages like "tensorflow". It can take some time to install all the required packages.)*

```bash
pip install -r requirements.txt
```

4. To execute the main code successfully, we need 2 pickle files. Because of the large file size, can't upload to github. Although you can get those files from [here](https://drive.google.com/drive/folders/1XdPAhN9saOuDnvxaLa21aeq3ncmllrOA?usp=sharing). Download and keep that files under the main directory like the below folder structure.

```bash
- Fashion-Recommender-System
    |-- fashion-dataset
    |   |-- fashion-dataset
    |   |-- images
    |   |-- styles
    |   |-- images.csv
    |   |-- styles.csv
    |
    |-- features.pkl
    |-- filenames.pkl
    |-- utilities.py
    |-- test.py
    |-- main.py
    |-- README.md
    |-- .gitignore
```

Or you can create those 2 files by executing the below command into your terminal. This will take long time to finish the execution process.

```bash
python utilities.py
```

5. Now we are ready good to go. Execute the below command to your terminal.

```bash
streamlit run main.py
```

This will start an server to your local machine. Open the link to your browser. This should like this below image

![Home Page](/project_images/image-1.png)

By clicking `Browse files` you can choose some test images from the `test-images` directory or can upload completly new fashion image. If everything goes well, can see a top 5 recommender images like this.

![Demo Project](/project_images/image-2.png)

Where the big image is your uploaded image and the below 5 images are the recommender images that are stored to your image dataset.

## ✏ Next Challenges:
1. Here we have used around 45k fashion images. But in an e-commerce sites, there are images in terms of millions. If we use those losts of images, the client can fase long time loading waiting period. According to the implementation, we can use [Annoy](https://github.com/spotify/annoy) library build by Spotify or can find out some other techniques to solve this issue.
2. Next is to do the job with the millions of images because it can take lots of hard disk space. For this it is also hard to deplopy and maintain this project. For that we can use [AWS](https://aws.amazon.com) S3 bucket or other like this service providers.
3. Create more flexible and fast code that can reduce the Time/Space complexity.