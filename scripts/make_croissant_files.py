import mlcroissant as mlc
import json
from pathlib import Path

def make_croissant_files():
    '''Programmatically creates croissant metadata files for each datasets'''
    
    # ----------------------------------------------------------
    # TEM1 Dataset
    # ----------------------------------------------------------
    tem1_dist = [
        mlc.FileObject(
            id="github-repository",
            name="astih-github-repository",
            description="Full ASTIH collection repository on GitHub.",
            content_url="https://github.com/axondeepseg/ASTIH",
            encoding_formats=["git+https"],
            sha256='main'
        ),
        mlc.FileObject(
            id="tem1-dandi-archive",
            name="TEM1_dataset",
            description="TEM Images of Corpus Callosum in Control and Cuprizone-Intoxicated Mice with Axon and Myelin Segmentations",
            content_url="https://dandiarchive.org/dandiset/001436/0.250512.1625",
            encoding_formats=["https"],
            sha256="6268e22a3ed7dda92a8ecf75bdef370027c00bb2970489d4d4f7c3f71b1afca6",
        ),
        mlc.FileSet(
            id="tem1-image-fileset",
            name="tem1-images",
            description="Grayscale TEM images (PNG)",
            encoding_formats=["image/png"],
            includes="TEM1_dataset/sub-*/micr/*.png",
        ),
        mlc.FileSet(
            id="tem1-axon-mask-fileset",
            name="tem1-axon-masks",
            description="Axon masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="TEM1_dataset/derivatives/labels/sub-*/micr/*_seg-axon-manual.png",
        ),
        mlc.FileSet(
            id="tem1-myelin-mask-fileset",
            name="tem1-myelin-masks",
            description="Myelin masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="TEM1_dataset/derivatives/labels/sub-*/micr/*_seg-myelin-manual.png",
        ),
        mlc.FileObject(
            id="tem1-readme",
            name="tem1-readme",
            description="Detailed description file for the TEM1 dataset.",
            encoding_formats=["text/markdown"],
            contained_in=["tem1-dandi-archive"],
            content_url="README",
        ),
        mlc.FileObject(
            id="tem1-sample-file",
            name="tem1-sample-file",
            description="TSV file listing all available samples in the dataset.",
            encoding_formats=["text/tab-separated-values"],
            contained_in=["tem1-dandi-archive"],
            content_url="samples.tsv",
        ),
        mlc.FileObject(
            id="tem1-participant-file",
            name="tem1-participant-file",
            description="TSV file listing all participants in the dataset.",
            encoding_formats=["text/tab-separated-values"],
            contained_in=["tem1-dandi-archive"],
            content_url="participants.tsv",
        )
    ]
    tem1_metadata = mlc.Metadata(
        name="TEM1",
        description=(
            "TEM dataset for AxonDeepSeg (https://axondeepseg.readthedocs.io/) 158 brain "
            "(splenium) samples from 20 mice with axon and myelin manual segmentation labels. "
            "Every individual axon is separated from neighboring fibers with a 2 pixel delineation, "
            "such that this semantic segmentation dataset also doubles as an instance segmentation "
            "dataset. In our original paper (Zaimi et al. 2018), the FOV was reported to be 6x9 um^2. "
            "This is because 1) these original values were reported in the original data reference "
            "(below), and 2) our images here are slightly cropped at the bottom relative to the "
            "original data in order to remove the scale bar. Our original paper (Zaimi et al. 2018) "
            "reported the resolution as being 0.002 micrometer, which was (for an unknown reason) "
            "rounded in the paper from the true value of 0.00236 micrometer, as reported in the "
            "original data reference (below). "
            "Reference for the origin of the data: Jelescu, I. O. et al. In vivo quantification "
            "of demyelination and recovery using compartment-specific diffusion MRI metrics validated "
            "by electron microscopy. Neuroimage 132, 104–114 (2016). See "
            "https://doi.org/10.1016/j.neuroimage.2016.02.004 (Center for Biomedical Imaging, "
            "Department of Radiology, New York University School of Medicine, New York, NY, USA) "
            "The original aim of the 2016 study was to quantify demyelination in mice intoxicated "
            "with cuprizone in both accute (6 weeks) and chronic (12 weeks) scenarios. As such, "
            "this dataset includes samples from both healthy and intoxicated mouse groups."
        ),
        cite_as=(
            "@misc{https://doi.org/10.48324/dandi.001436/0.250512.1625, doi = {10.48324/DANDI.001436/0.250512.1625}, "
            "url = {https://dandiarchive.org/dandiset/001436/0.250512.1625}, author = {Jelescu,  Ileana and Fieremans,  "
            "Els and Collin,  Armand and Cohen-Adad,  Julien}, title = {TEM Images of  Corpus Callosum in Control and "
            "Cuprizone-Intoxicated Mice with Axon and Myelin Segmentations}, publisher = {DANDI Archive}, year = {2025}}"
        ),
        license="CC-BY-4.0",
        distribution=tem1_dist,
        keywords=["axon", "myelin", "TEM", "microscopy", "segmentation"],
        url="https://doi.org/10.48324/dandi.001436/0.250512.1625",
        date_published="2025-05-12",
    )
    print('TEM1 croissant file generation\n', tem1_metadata.issues.report())
    with open('data/croissant_tem1.json', 'w') as f:
        content = tem1_metadata.to_json()
        content = json.dumps(content, indent=2, default=str)
        f.write(content)
        f.write('\n')
    print('TEM1 croissant file succesfully generated.')    

    # ----------------------------------------------------------
    # TEM2 Dataset
    # ----------------------------------------------------------
    tem2_dist = [
        mlc.FileObject(
            id="github-repository",
            name="astih-github-repository",
            description="Full ASTIH collection repository on GitHub.",
            content_url="https://github.com/axondeepseg/ASTIH",
            encoding_formats=["git+https"],
            sha256='main'
        ),
        mlc.FileObject(
            id="tem2-dandi-archive",
            name="TEM2_dataset",
            description=" TEM Images of Corpus Callosum in Flox/SRF-cKO Mice",
            content_url="https://doi.org/10.48324/dandi.001350/0.250511.1527",
            encoding_formats=["https"],
            sha256="b156054f4a0626e69f6275f3a873e05f7a25975ef761ac63ae6f3fa77fd38f11",
        ),
        mlc.FileSet(
            id="tem2-image-fileset",
            name="tem2-images",
            description="Grayscale TEM images (PNG)",
            encoding_formats=["image/png"],
            includes="TEM2_dataset/sub-*/micr/*.png",
        ),
        mlc.FileSet(
            id="tem2-axon-mask-fileset",
            name="tem2-axon-masks",
            description="Axon masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="TEM2_dataset/derivatives/labels/sub-*/micr/*_seg-axon-manual.png",
        ),
        mlc.FileSet(
            id="tem2-uaxon-mask-fileset",
            name="tem2-uaxon-masks",
            description="Unmyelinated axon masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="TEM2_dataset/derivatives/labels/sub-*/micr/*_seg-uaxon-manual.png",
        ),
        mlc.FileSet(
            id="tem2-myelin-mask-fileset",
            name="tem2-myelin-masks",
            description="Myelin masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="TEM2_dataset/derivatives/labels/sub-*/micr/*_seg-myelin-manual.png",
        ),
        mlc.FileSet(
            id="tem2-nuclei-mask-fileset",
            name="tem2-nuclei-masks",
            description="Oligodendrocyte nucleus masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="TEM2_dataset/derivatives/labels/sub-*/micr/*_seg-nuclei-manual.png",
        ),
        mlc.FileSet(
            id="tem2-process-mask-fileset",
            name="tem2-process-masks",
            description="Oligodendrocyte process masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="TEM2_dataset/derivatives/labels/sub-*/micr/*_seg-process-manual.png",
        ),
        mlc.FileObject(
            id="tem2-readme",
            name="tem2-readme",
            description="Detailed description file for the TEM2 dataset.",
            encoding_formats=["text/markdown"],
            contained_in=["tem2-dandi-archive"],
            content_url="README",
        ),
        mlc.FileObject(
            id="tem2-sample-file",
            name="tem2-sample-file",
            description="TSV file listing all available samples in the dataset.",
            encoding_formats=["text/tab-separated-values"],
            contained_in=["tem2-dandi-archive"],
            content_url="samples.tsv",
        ),
        mlc.FileObject(
            id="tem2-participant-file",
            name="tem2-participant-file",
            description="TSV file listing all participants in the dataset.",
            encoding_formats=["text/tab-separated-values"],
            contained_in=["tem2-dandi-archive"],
            content_url="participants.tsv",
        )
    ]
    tem2_metadata = mlc.Metadata(
        name="TEM2",
        description=(
            "This dataset contains 8000x transmission EM images of the Corpus Callosum in "
            "two groups of mice. The main goal of this project is to study unmyelinated axons. "
            "The technique used to prepare the samples is high pressure freezing and freeze "
            "substitution. This results in high-definition images where unmyelinated axons are "
            "remarkably visible. Imaging was done at UC Berkeley. This data was originally used "
            "in https://www.pnas.org/doi/10.1073/pnas.2307250121. The two mice groups are Wild "
            "Type (WT) and SRF conditional knockout (KO). There are 5 animals per group for a "
            "total of 10 and multiple images per mouse. Original magnification: 8000x This "
            "dataset includes segmentation masks for the following semantic classes: (myelinated) "
            "axon, (unmyelinated) axon, myelin, oligodendrocyte nucleus and oligodendrocyte processes. "
            "Further details on a deep segmentation model trained on this dataset can be found "
            "at https://github.com/axondeepseg/model_seg_unmyelinated_tem"
        ),
        cite_as=(
            "@misc{https://doi.org/10.48324/dandi.001350/0.250511.1527, doi = "
            "{10.48324/DANDI.001350/0.250511.1527}, url = "
            "{https://dandiarchive.org/dandiset/001350/0.250511.1527}, author = {Collin,  "
            "Armand and Soulaire,  Tom}, keywords = {TEM,  Electron Microscopy,  Histology,  "
            "CNS,  SRF,  axon,  myelin,  segmentation}, title = {TEM Images of Corpus "
            "Callosum in Flox/SRF-cKO Mice}, publisher = {DANDI Archive}, year = {2025}}"
        ),
        license="CC-BY-4.0",
        distribution=tem2_dist,
        keywords=["axon", "myelin", "unmyelinated", "TEM", "microscopy", "segmentation"],
        url="https://doi.org/10.48324/dandi.001350/0.250511.1527",
        date_published="2025-05-11",
    )
    print('TEM2 croissant file generation\n', tem2_metadata.issues.report())
    with open('data/croissant_tem2.json', 'w') as f:
        content = tem2_metadata.to_json()
        content = json.dumps(content, indent=2, default=str)
        f.write(content)
        f.write('\n')
    print('TEM2 croissant file succesfully generated.')

    # ----------------------------------------------------------
    # SEM1 Dataset
    # ----------------------------------------------------------
    sem1_dist = [
        mlc.FileObject(
            id="github-repository",
            name="astih-github-repository",
            description="Full ASTIH collection repository on GitHub.",
            content_url="https://github.com/axondeepseg/ASTIH",
            encoding_formats=["git+https"],
            sha256='main'
        ),
        mlc.FileObject(
            id="sem1-dandi-archive",
            name="SEM1_dataset",
            description="SEM Images of Rat Spinal Cord with Axon and Myelin Segmentations",
            content_url="https://doi.org/10.48324/dandi.001442/0.250512.1626",
            encoding_formats=["https"],
            sha256="917c3eed9329d90674b4f5e62d7e12eed4c54bad46b414a441eccc72cea80646",
        ),
        mlc.FileSet(
            id="sem1-image-fileset",
            name="sem1-images",
            description="Grayscale SEM images (PNG)",
            encoding_formats=["image/png"],
            includes="SEM1_dataset/sub-*/micr/*.png",
        ),
        mlc.FileSet(
            id="sem1-axon-mask-fileset",
            name="sem1-axon-masks",
            description="Axon masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="SEM1_dataset/derivatives/labels/sub-*/micr/*_seg-axon-manual.png",
        ),
        mlc.FileSet(
            id="sem1-myelin-mask-fileset",
            name="sem1-myelin-masks",
            description="Myelin masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="SEM1_dataset/derivatives/labels/sub-*/micr/*_seg-myelin-manual.png",
        ),
        mlc.FileObject(
            id="sem1-readme",
            name="sem1-readme",
            description="Detailed description file for the SEM1 dataset.",
            encoding_formats=["text/markdown"],
            contained_in=["sem1-dandi-archive"],
            content_url="README",
        ),
        mlc.FileObject(
            id="sem1-sample-file",
            name="sem1-sample-file",
            description="TSV file listing all available samples in the dataset.",
            encoding_formats=["text/tab-separated-values"],
            contained_in=["sem1-dandi-archive"],
            content_url="samples.tsv",
        ),
        mlc.FileObject(
            id="sem1-participant-file",
            name="sem1-participant-file",
            description="TSV file listing all participants in the dataset.",
            encoding_formats=["text/tab-separated-values"],
            contained_in=["sem1-dandi-archive"],
            content_url="participants.tsv",
        )
    ]
    sem1_metadata = mlc.Metadata(
        name="SEM1",
        description=(
            "SEM dataset for AxonDeepSeg (https://axondeepseg.readthedocs.io/). 10 rat spinal "
            "cord samples (cervical level) with axon and myelin manual segmentation labels. "
            "Isotropic pixel size resolution ranging from 0.05 to 0.18 um. Every touching fiber "
            "is separated by a 1 pixel delineation in the masks. This dataset comprises "
            "acquisitions conducted by various researchers between 2015 and 2017. The images "
            "represent small cropped sections extracted from larger mosaic images."
        ),
        cite_as=(
            "@misc{https://doi.org/10.48324/dandi.001442/0.250512.1626, doi = "
            "{10.48324/DANDI.001442/0.250512.1626}, url = "
            "{https://dandiarchive.org/dandiset/001442/0.250512.1626}, author = {Saliani,  "
            "Ariane and Duval,  Tanguy and Nami,  Harris and Husein,  Nafisa and Collin,  "
            "Armand and Zaimi,  Aldo and Bourget,  Marie-Hélène and Cohen-Adad,  Julien}, "
            "title = {SEM Images of Rat Spinal Cord with Axon and Myelin Segmentations}, "
            "publisher = {DANDI Archive}, year = {2025}}"
        ),
        license="CC-BY-4.0",
        distribution=sem1_dist,
        keywords=["axon", "myelin", "SEM", "microscopy", "segmentation"],
        url="https://doi.org/10.48324/dandi.001442/0.250512.1626",
        date_published="2025-05-12",
    )
    print('SEM1 croissant file generation\n', sem1_metadata.issues.report())
    with open('data/croissant_sem1.json', 'w') as f:
        content = sem1_metadata.to_json()
        content = json.dumps(content, indent=2, default=str)
        f.write(content)
        f.write('\n')
    print('SEM1 croissant file succesfully generated.')

    # ----------------------------------------------------------
    # BF1 Dataset
    # ----------------------------------------------------------
    bf1_dist = [
        mlc.FileObject(
            id="github-repository",
            name="astih-github-repository",
            description="Full ASTIH collection repository on GitHub.",
            content_url="https://github.com/axondeepseg/ASTIH",
            encoding_formats=["git+https"],
            sha256='main'
        ),
        mlc.FileObject(
            id="bf1-dandi-archive",
            name="BF1_dataset",
            description="BF Images of Rat Nerves at Different Regeneration Stages",
            content_url="https://doi.org/10.48324/dandi.001440/0.250509.1913",
            encoding_formats=["https"],
            sha256="ee678e640ce9d5f09373ad91c2f54661a41c434c8ae925914ce26f235e830db5",
        ),
        mlc.FileSet(
            id="bf1-image-fileset",
            name="bf1-images",
            description="Grayscale BF images (PNG)",
            encoding_formats=["image/png"],
            includes="BF1_dataset/sub-*/micr/*.png",
        ),
        mlc.FileSet(
            id="bf1-axon-mask-fileset",
            name="bf1-axon-masks",
            description="Axon masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="BF1_dataset/derivatives/labels/sub-*/micr/*_seg-axon-manual.png",
        ),
        mlc.FileSet(
            id="bf1-myelin-mask-fileset",
            name="bf1-myelin-masks",
            description="Myelin masks in grayscale, 8-bit PNG format.",
            encoding_formats=["image/png"],
            includes="BF1_dataset/derivatives/labels/sub-*/micr/*_seg-myelin-manual.png",
        ),
        mlc.FileObject(
            id="bf1-readme",
            name="bf1-readme",
            description="Detailed description file for the BF1 dataset.",
            encoding_formats=["text/markdown"],
            contained_in=["bf1-dandi-archive"],
            content_url="README",
        ),
        mlc.FileObject(
            id="bf1-sample-file",
            name="bf1-sample-file",
            description="TSV file listing all available samples in the dataset.",
            encoding_formats=["text/tab-separated-values"],
            contained_in=["bf1-dandi-archive"],
            content_url="samples.tsv",
        ),
        mlc.FileObject(
            id="bf1-participant-file",
            name="bf1-participant-file",
            description="TSV file listing all participants in the dataset.",
            encoding_formats=["text/tab-separated-values"],
            contained_in=["bf1-dandi-archive"],
            content_url="participants.tsv",
        )
    ]
    bf1_metadata = mlc.Metadata(
        name="BF1",
        description=(
            "Bright-Field Optical Microscopy (BF) dataset for AxonDeepSeg "
            "(https://axondeepseg.readthedocs.io/). Rat peripheral nerves across different "
            "axonal regeneration stages. Data collected from adult rats in experimental nerve "
            "repair studies. 8 samples cropped to ROI used as training data. Corresponding "
            "axon, myelin, axonmyelin manual segmentation labels in derivatives. This dataset "
            "is a subset of the full dataset used in this publication: "
            "https://www.nature.com/articles/s41598-022-10066-6. Training set for the "
            "AxonDeepSeg model developed and validated in the aforementioned article."
        ),
        cite_as=(
            "@misc{https://doi.org/10.48324/dandi.001440/0.250509.1913, doi = "
            "{10.48324/DANDI.001440/0.250509.1913}, url = "
            "{https://dandiarchive.org/dandiset/001440/0.250509.1913}, author = {Daeschler, "
            "Simeon and Bourget,  Marie-Hélène and Cohen-Adad,  Julien and Borschel,  Gregory "
            "Howard}, title = {Bright-Field Images of Rat Nerves at Different Regeneration "
            "Stages with Axon and Myelin Segmentations}, publisher = {DANDI Archive}, year = {2025}}"
        ),
        license="CC-BY-4.0",
        distribution=bf1_dist,
        keywords=["axon", "myelin", "BF", "Bright-Field", "microscopy", "segmentation"],
        url="https://doi.org/10.48324/dandi.001440/0.250509.1913",
        date_published="2025-05-09",
    )
    print('BF1 croissant file generation\n', bf1_metadata.issues.report())
    with open('data/croissant_bf1.json', 'w') as f:
        content = bf1_metadata.to_json()
        content = json.dumps(content, indent=2, default=str)
        f.write(content)
        f.write('\n')
    print('BF1 croissant file succesfully generated.')

if __name__ == "__main__":
    assert (Path('.') / 'data').exists(), "Please run this script from the root of the repository."
    make_croissant_files()