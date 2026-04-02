# Before building, run model/train.py and copy artifacts:
#   cd model && python train.py
#   cp -r artifacts/ ../consumer/model/
#   cd ../consumer && docker build -t consumer .

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY autoencoder.py consumer.py ./

# Model artifacts baked into image at /model
COPY model/ /model/

CMD ["python", "consumer.py"]
