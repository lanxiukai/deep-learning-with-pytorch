"""Named lesson downloads and their repository data-directory layout."""

DATA_URL = "https://d2l-data.s3-accelerate.amazonaws.com/"


DATA_HUB = {
    "kaggle_house_train": (
        DATA_URL + "kaggle_house_pred_train.csv",
        "585e9cc93e70b39160e7921475f9bcd7d31219ce",
    ),
    "kaggle_house_test": (
        DATA_URL + "kaggle_house_pred_test.csv",
        "fa19780a7b011d9b009e8bff8e99922a8ee2eb90",
    ),
    "time_machine": (
        DATA_URL + "timemachine.txt",
        "090b5e7e70c295757f55df93cb0a180b9691891a",
    ),
    "fra-eng": (
        DATA_URL + "fra-eng.zip",
        "94646ad1522d915e7b0f9296181140edcf86a4f5",
    ),
    "pokemon": (
        DATA_URL + "pokemon.zip",
        "c065c0e2593b8b161a2d7873e42418bf6a21106c",
    ),
    "airfoil": (
        DATA_URL + "airfoil_self_noise.dat",
        "76e5be1548fd8222e5074cf0faae75edff8cf93f",
    ),
}


DOWNLOAD_SUBDIRECTORIES = {
    "airfoil": "airfoil_self_noise",
    "kaggle_house_train": "kaggle_house_price",
    "kaggle_house_test": "kaggle_house_price",
    "time_machine": "time_machine",
    "fra-eng": "fra_eng",
    "pokemon": "pokemon",
}
