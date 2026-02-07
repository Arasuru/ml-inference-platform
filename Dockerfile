#os and python version
FROM python:3.11-slim

#working directory: will be created if it doesn't exist
WORKDIR /app

#copy the requirements file to the working directory
COPY requirements.txt .

#install the dependencies specified in the requirements file, using pip with the --no-cache-dir option to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

#copy the rest of the application code to the working directory
COPY . .

#expose the port that the application will run on
EXPOSE 8000

#command to run the application using uvicorn, specifying the host and port
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]