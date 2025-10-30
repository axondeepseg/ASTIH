# 🚧 Expanding ASTIH
This initiative was meant to be easily expanded upon, but data harmonization should be prioritized. This guide will describe how to add a dataset to the collection.

## ☕ Prerequisites
First, make sure your dataset is in the correct format (BIDS) and passes the [validator](https://bids-standard.github.io/legacy-validator/). The README should contain context information and a brief overview of the data and its origin. Also, make sure that the ground-truth segmentation masks have the expected suffixes, i.e.

- `_seg-axon-manual` for the axon class
- `_seg-myelin-manual` for the myelin class
- `_seg-uaxon-manual` for the unmyelinated axon class

Finally, a pretrained segmentation model should be provided along the data (for reference: check the latest release in [this repository](https://github.com/axondeepseg/model_seg_unmyelinated_tem/tree/main)). Ideally, you should create a new issue here (https://github.com/axondeepseg/ASTIH/issues) so that we can review the suggested contribution.

## 🧠 Uploading to DANDI
This is where we host the datasets. The [DANDI archive](https://dandiarchive.org/) is an platform to share and publish neurophysiology data. DANDI also provides a nice interface to upload and retrieve the data. To upload data, you need to register in order to receive an API key. Detailed instructions are available [here](https://docs.dandiarchive.org/user-guide-sharing/creating-dandiset/), but overall these are the steps:

1. Create a new dandiset using the DANDI web application. You will need to specify a title, a description and a license. For the title, please be consistent with out [other datasets](https://dandiarchive.org/dandiset/search?search=axondeepseg). For the description, go for something similar to the dataset README, and ideally even more concise. This will **register** the dandiset, but we still need to add the data!
2. You now have a webpage for your dandiset. Go ahead and edit the metadata (click on _Metadata_ on the right). Most importantly, you should put some keywords in **General > Study Target**. The second crucial part is the section **Dandiset contributors**. This is where you enumerate a comprehensive list of authors for the dataset, which will also be used when the dataset is cited. Also, note that this list is separate from the "Owners" of the data, which can include non-contributors.

TO BE CONTINUED
