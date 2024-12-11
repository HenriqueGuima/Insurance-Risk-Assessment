# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /

# Copy the project files into the container
COPY . /

# Install system dependencies for Rust and Cargo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y

# Add Cargo to PATH for all future commands
ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --upgrade pip

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter


# Define the command to run the scraper
# CMD ["python", "supervised_learning_insurance.ipnyb"]
CMD ["jupyter", "nbconvert", "--to", "script", "download_dataset.ipynb", "--execute"]

