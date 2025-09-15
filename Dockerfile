# Use an official PyTorch image as a base
FROM quay.io/modh/training:py311-cuda124-torch251

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy your fine-tuning script into the container
COPY finetune.py .

# Command to execute when the container starts
CMD ["python", "finetune.py"]