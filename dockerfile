FROM python:3.10.11

# Create a Working Directory within a your container for your application
WORKDIR /app

# Copy the content of my requirements.txt into a temporary dictionary in the container
COPY requirements.txt /tmp/requirements.txt

# Install packages in the requirements.txt file
RUN python -m pip install --timeout 300000 -r /tmp/requirements.txt

# Copy all files and folders into the container's working directory
COPY . /app/

# Expose port 8077 outside the container
EXPOSE 8077

# Run the FastAPI application
CMD ["uvicorn", "Dev.Home:app", "--host", "0.0.0.0", "--port", "8077"]