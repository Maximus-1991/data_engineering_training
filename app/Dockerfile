FROM python:3.7-slim
WORKDIR /usr/local/airflow
ENV FLASK_APP "dags/app.py"
ENV FLASK_RUN_HOST 0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
WORKDIR "/usr/local/airflow/dags"
CMD ["flask", "run"]