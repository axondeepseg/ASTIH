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
To notify the *get_data.py* script that you added a new dataset, you should edit the following file: `scripts/dataset_list.json`. Create a new entry, like so:
```json
  {
    "name": "BF1",
    "id": "001440",
    "url": "https://dandiarchive.org/dandiset/001440/0.250509.1913",
    "desc": "BF Images of Rat Nerves at Different Regeneration Stages with Axon and Myelin Segmentations",
    "test_set": [
        "sub-uoftRat02",
        "sub-uoftRat07"
    ],
    "model_url": "https://github.com/axondeepseg/default-BF-model/releases/download/r20240405/model_seg_rat_axon-myelin_bf_light.zip"
  }
```
For the model URL, be sure to put the link to the released ZIP so that the pipeline can automatically download it later. The `id` field should be the one assigned by DANDI, and the `url` field should contain the dataset DOI. 

Finally, please note that the `test_set` field accepts 2 different types:
1. A **string**: in this case, the value is interpreted as the _URL of an independently-hosted test set_ (external test set). For example, refer to the TEM2 dataset.
2. A **list**: here, the value is interpreted as a _list of subjects_ (internal test set). These subjects will be used as a test set. This is the default method.

That's it! Your new dataset is now part of the collection. If everything went according to plan, users should now be able to download your dataset and the associated segmentation model, and evaluate the latter on the test set.

## ☔ (Optional) Update ASTIH splash page
If you feel like it, you can also update the ASTIH splash page so that your new dataset becomes visible there! To do so, edit the `docs/index.html` file and add an _Image card_. There is a div that contains a block for every dataset. To add yours, simply add a codeblock (see example below). Note that you don't need to care about placement: the way it is currently set up, all blocks appear in a grid with 3 columns, and the HTML should take care of placing your new block in the correct location.

Here is what your new block should look like (modify the text in upper case)
```html
<!-- Image Card X+1 -->
<a href="DOI_OF_YOUR_NEW_DATASET" class="block glass-panel rounded-xl overflow-hidden glass-card-hover transition-all duration-300 group">
  <div class="h-48 bg-slate-800/50 w-full overflow-hidden relative">

    <img src="imgs/SMALL_SQUARE_IMAGE_FOR_YOUR_DATASET" alt="ALT_TEXT" class="w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
      onerror="this.src='https://placehold.co/600x400/1e293b/475569?text=Image+Missing';">
    <div class="absolute bottom-2 left-2 px-2 py-1 bg-black/60 rounded text-xs text-white backdrop-blur-sm">MODALITY</div>
  </div>
  <div class="p-4 text-left">
    <div class="flex justify-between items-center"
      <h4 class="font-medium text-slate-200">SHORT_NAME_OF_THE_DATASET</h4>
      <i data-lucide="external-link" class="w-4 h-4 text-slate-500 group-hover:text-indigo-400 transition-colors"></i>
    </div>
    <p class="text-xs text-slate-500 mt-1">SHORT_SUMMARY_OF_THE_DATASET</p>
  </div>
</a>
```
