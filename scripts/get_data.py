from pathlib import Path
from dandi.download import download


class ASTIHDataset:
    def __init__(self, name, id, url, desc, test_set, model_url):

        self.name = name
        self.dandi_id = id
        self.url = url
        self.desc = desc
        if isinstance(test_set, list):
            self.test_set_type = 'internal'
            self.test_subjects = test_set
        elif type(test_set) is str:
            self.test_set_type = 'external'
            self.test_set_url = test_set
        else:
            raise ValueError("test_set must be a list of subjects or a URL")
        self.model_release = model_url

DATASETS = [
    ASTIHDataset(
        name="TEM1",
        id="001436",
        url="https://dandiarchive.org/dandiset/001436/0.250512.1625",
        desc="TEM Images of Corpus Callosum in Control and Cuprizone-Intoxicated Mice with Axon and Myelin Segmentations",
        test_set=[
            "sub-nyuMouse26"
        ],
        model_url="https://github.com/axondeepseg/default-TEM-model/releases/download/r20240403/model_seg_mouse_axon-myelin_tem_light.zip"
    ),
    ASTIHDataset(
        name="TEM2",
        id="001350",
        url="https://dandiarchive.org/dandiset/001350/0.250511.1527",
        desc="TEM Images of Corpus Callosum in Flox/SRF-cKO Mice",
        test_set="https://github.com/axondeepseg/data_axondeepseg_srf_testing/archive/refs/tags/r20250513-neurips2025.zip",
        model_url="https://github.com/axondeepseg/model_seg_unmyelinated_tem/releases/download/r20240708-stanford/model_seg_unmyelinated_stanford_tem_best.zip"
    ),
    ASTIHDataset(
        name="SEM1",
        id="001442",
        url="https://dandiarchive.org/dandiset/001442/0.250512.1626",
        desc="SEM Images of Rat Spinal Cord with Axon and Myelin Segmentations with Myelinated and Unmyelinated Axon Segmentations",
        test_set=[
            "sub-rat6"
        ],
        model_url="https://github.com/axondeepseg/default-SEM-model/releases/download/r20240403/model_seg_rat_axon-myelin_sem_light.zip"
    ),
    ASTIHDataset(
        name="BF1",
        id="001440",
        url="https://dandiarchive.org/dandiset/001440/0.250509.1913",
        desc="BF Images of Rat Nerves at Different Regeneration Stages with Axon and Myelin Segmentations",
        test_set=[
            "sub-rat2",
            "sub-rat7"
        ],
        model_url="https://github.com/axondeepseg/default-BF-model/releases/download/r20240405/model_seg_rat_axon-myelin_bf_light.zip"
    )
]

def main():
    # Create a directory to store the downloaded data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # download dandisets
    urls = [dataset.url for dataset in DATASETS]
    download(urls, data_dir)


if __name__ == "__main__":
    main()