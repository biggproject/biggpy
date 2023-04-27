FROM gitlab.ilabt.imec.be/pilot/dependency_proxy/containers/python:3.10-slim

# Install Python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY /src /src
COPY pyproject.toml .
COPY config.yml .
WORKDIR /

RUN pip3 install poetry
RUN poetry config http-basic.glpackages package-registry CeygpzDyFvQ4xemq6yp4 && poetry config virtualenvs.create false && poetry install
RUN poe force-cuda11

EXPOSE 5000
ENV FLASK_APP=src.backend.main

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]     
