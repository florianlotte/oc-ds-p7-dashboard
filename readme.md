# OPENCLASSROOMS - Data Scientist

## Projet 7 - Dashboard

Frontend et backend du Dashboard pour le projet 7 openclassrooms data scientist

<p>
  <img src="https://img.shields.io/badge/python-%3E%3D3.9-green" alt="Supported Python versions">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

### How to run

Localy :

```console
python ./app_api.py
python ./app_dashboard.py
```

With docker :

```console
docker build -f Dockerfile_api -t p07-api:latest .
docker run -p8889:80 p07-api

docker build -f Dockerfile_dashboard -t p07-dashboard:latest .
docker run -p8888:80 p07-dashboard
```

With composer :

```console
docker-compose up
```

### How to use

The swagger and the doc are available at:
* **swagger** [http://127.0.0.1:8889/docs](http://localhost:8889/docs)
* **redoc** [http://127.0.0.1:8889/redoc](http://localhost:8889/redoc)
* **dashboard** [http://127.0.0.1:8888](http://localhost:8888)
