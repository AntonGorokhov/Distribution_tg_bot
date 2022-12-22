FROM python:3.10

COPY . app

EXPOSE 8888

RUN pip install -r app/requirements.txt

CMD ["python", "app/main.py"]