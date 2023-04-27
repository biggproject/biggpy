FROM gitlab.ilabt.imec.be/pilot/dependency_proxy/containers/python:3.10-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc &&  \
    apt clean &&  \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /reports && \
    mkdir /reports/wandb

WORKDIR /

COPY /rltraining /rltraining
COPY /configuration_files /configuration_files
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY train.py train.py

RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "train.py"]
