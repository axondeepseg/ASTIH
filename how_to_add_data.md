# 🚧 Expanding ASTIH
This initiative was meant to be easily expanded upon, but data harmonization should be prioritized. This guide will describe how to add a dataset to the collection.

## ☕ Prerequisites
First, make sure your dataset is in the correct format (BIDS) and passes the [validator](https://bids-standard.github.io/legacy-validator/). The README should contain context information and a brief overview of the data and its origin. Also, make sure that the ground-truth segmentation masks have the expected suffixes, i.e.

- `_seg-axon-manual` for the axon class
- `_seg-myelin-manual` for the myelin class
- `_seg-uaxon-manual` for the unmyelinated axon class

Finally, a pretrained segmentation model should be provided along the data (for reference: check the latest release in [this repository](https://github.com/axondeepseg/model_seg_unmyelinated_tem/tree/main)). Ideally, you should create a new issue here (https://github.com/axondeepseg/ASTIH/issues) so that we can review the suggested contribution.

## 🧠 Uploading to DANDI
This is where we host the datasets. The [DANDI archive](https://dandiarchive.org/) is a platform to share and publish neurophysiology data. DANDI also provides a nice interface to upload and retrieve the data. To upload a dataset, you need to register to receive an API key. Your DANDI account will be linked to both your github account and your academic email. 

To upload your data, detailed instructions are available [in the doc](https://docs.dandiarchive.org/user-guide-sharing/creating-dandiset/), but overall these are the steps:

1. Create a new dandiset using the DANDI web application. You will need to specify a title, a description and a license. For the title, please be consistent with our [other datasets](https://dandiarchive.org/dandiset/search?search=axondeepseg). For the description, go for something similar to the dataset README, and ideally even more concise. This will **register** the dandiset, but we still need to add the data!
2. You now have a webpage for your dandiset. Go ahead and edit the metadata (click on _Metadata_ on the right, see screenshot below). Most importantly, you should put some keywords in **General > Study Target**. The second crucial part is the section **Dandiset contributors**. This is where you enumerate a comprehensive list of authors for the dataset, which will also be used when the dataset is cited. Also, note that this list is separate from the "Owners" of the data, which can include non-contributors. <center> <img width="326" height="290" alt="Screenshot_20251030_135809" src="https://github.com/user-attachments/assets/044de92c-7750-4a89-812c-43b4ed012f7a" /> </center>
4. Edit owners. In the future, you might not be an appropriate contact person/owner for the data (for example if you find an awesome job as an astronaut after your PhD). To ensure someone has the keys to the data while you reach for the stars, you should at least add `jcohenadad` as an owner, and possibly `mathieuboudreau` as well.
5. Now is time to add the data itself. The procedure is detailed in [this doc section](https://docs.dandiarchive.org/user-guide-sharing/uploading-data/). First, you need to install the dandi client (`pip install -U dandi`) and set up your API keys. Then, download the draft dandiset with
```
dandi download https://dandiarchive.org/dandiset/<dataset_id>/draft
cd <dataset_id>
```
Now, place the content of your valid BIDS dataset inside this folder. You can validate the data via
```
dandi validate .
```
Finally, simply upload the data with 
```
dandi upload
```

6. At this point, you should double check everything in the web application. When the dandiset is ready, click on _Publish_!
   <img width="300" height="30" alt="Screenshot_20251030_145558" src="https://github.com/user-attachments/assets/24bde507-d258-4806-bed2-b0195685c46a" />


## 🌉 Adding the dataset to ASTIH

TODO: add details
