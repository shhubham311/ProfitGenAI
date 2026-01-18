# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 7860

# Command to run the application using Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]