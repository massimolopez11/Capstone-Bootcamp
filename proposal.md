# Image Recognition from Photo Albums

## I. Introduction
The project aims to create a model for image recognition of photo albums.

## II. Project Objective
The project will entail creating an model for image recognition. The goal of the project is to identify the images from a photo album (Zip file), upload them to a website, and classify the faces in the photo album. Once the model is generated, a separate feature of the site would be to upload an image from the album and recognize the different faces from the album.
  
## III. Data Description
They will be images from my photo album. I have many photos on a storage device at home. They are photos that contain peoples' faces in them. They can vary from low resolution to higher resolution photos, color vs black and white photos, old vs new photos. Taking the images from my Google Drive/Google Photos might be more practical. This could be another feature to do later.

## IV. Methodology

The difficulty is knowing the type of model to use and how to implement it. Since the dataset consists of images of 1 or many people, then this would involve clusters. The general steps would then be something like the following:

1. Define features from the images (Using a library like `Img2Vec` from `pytorch`).
2. create clusters/centroids from the image features (Either `kmeans` or an alternative for better results).
3. The Website will have groups of images with the label (Or the image on display) being the first data point from each cluster. It is then up to the user to type that person’s name to specify what the images in the cluster are for.

## V. Expected Deliverables

The plan is to deliver a blueprint for generating the model since this would be universal for any photo album, not just my own. So, each user would generate a personalized model. The goal is to make this process efficient enough that it is seamless to do for any user while still having the least amount of error (elbow plot might help).

## VI. Timeline and Tasks
The following steps are a general idea for each task and the timeline needed for each task.

1. Setup repo: Have workflows for PR, CI automation for linting and code errors, and CD for deployment using `streamlit`, and install dependencies from `requirements.txt` if that is possible. [Week 1 (this week) Sept 2nd to 8th]
2. Gather the dataset: Images are on my local hard drive, so they must be uploaded to the server (or when developing, upload them to the repo) using `streamlit`. [Week 2 Sept 9th to 15th]
3. Generate features using `Img2Vec` from `pytorch` or something similar. [Week 2 Sept 9th to 15th]
4. Generate the model using `kmeans` or something similar. Report the error amount using an elbow plot and investigate other strategies based on those results. [Week 3 Sept 16th to 22nd]
5. Upload the clusters to `streamlit` so that the user can view the photos from each group. Could have a setting/toggle to either view the entire photo or just the data point (cropped image of the face for that cluster) [Week 3 Sept 16th to 22nd]
6. Upload an image to predict the person/people from the model [Week 3 Sept 16th to 22nd]
7. STRETCH: Social media side. Hold and click on images to upload them to Instagram. [Week 3 Sept 16th to 22nd]
8. STRETCH: Upload the images using another means (Google account via Google Photos, one drive, etc...) [Week 3 Sept 16th to 22nd]
9. Prepare boring presentation stuff [Week 4 Sept 23rd to 29th]
  
## VII. Potential Challenges

I've used clustering before in projects but not for images, so there might be issues creating an "accurate" model, which is subjective for clustering. Still, some issues might arise if the pictures lump many different people in the same clusters. The only solution to this is to change the type of unsupervised learning model and do some research on some better approaches. It would take some trial and error as well.

`streamlit` is a relatively new tool for me, along with the other tools like `Img2Vec`. Again, I need to research to see what I can do with these tools.