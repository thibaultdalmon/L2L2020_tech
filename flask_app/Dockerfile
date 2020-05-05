# Inherit from Python 3.6 image
FROM python:3.6

# Update pip, setuptools, and wheel to ensure that we have the latest versions.
RUN pip --no-cache-dir --disable-pip-version-check install --upgrade pip setuptools wheel

# Set a working directory
WORKDIR /usr/src

# Copy requirements
COPY requirements.txt .

# Install requirements
RUN pip install -r requirements.txt

# Create Data folders
RUN mkdir -p data_afo/test/cvkvp
RUN mkdir -p data_afo/val/cat
RUN mkdir -p data_afo/val/kanye
RUN mkdir -p data_afo/val/pikachu
RUN mkdir -p data_afo/train/cat
RUN mkdir -p data_afo/train/kanye
RUN mkdir -p data_afo/train/pikachu

# Copy current folder
COPY . .

# Run python code
CMD python app.py
