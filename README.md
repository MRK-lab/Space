# Space


# Sırasıyla işlemler

MODELE GÖRE GEREKLİ ALANLAR DÜZENLENMELİDİR (PROMT-TOKEN-REPO-ID)

1- Token geçerliliği kontrolü:

huggingface-cli login

-token gir


2- Yüklemeler:

sudo apt update

sudo apt install libcurl4-openssl-dev

pip3 install unsloth

pip3 install mistral-common


3- git reposunu yükleyelim:

git clone https://github.com/MRK-lab/Space.git


4- Token hatası alırsak upload_to_ht.py kodunu çalıştıracağız:

sudo apt install git-lfs

export HF_TOKEN="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"

python upload_to_hf.py -d ./mrkswe/<modelRepoAdı> -r mrkswe/<modelRepoAdı> --use-git-lfs
