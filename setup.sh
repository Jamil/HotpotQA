#!/usr/bin/env bash
aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300 s3 cp s3://jdhanani/libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb .
aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300 s3 cp s3://jdhanani/libcudnn7-dev.deb .
aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300 s3 cp s3://jdhanani/libcudnn7-doc_7.3.1.20-1+cuda9.0_amd64.deb .
sudo dpkg -i libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev.deb
sudo dpkg -i libcudnn7-doc_7.3.1.20-1+cuda9.0_amd64.deb

pip install --upgrade pip
pip install turibolt --index=https://pypi.apple.com/simple
pip uninstall -y tensorflow
pip install -U tensorflow-gpu==1.11.0
pip install joblib
./download.sh

source activate PY3
conda install pytorch=0.3.0 cuda90 -c pytorch
conda install spacy
