FROM python:3.12.1-slim-bullseye

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
