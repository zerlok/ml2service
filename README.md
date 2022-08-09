# ml2service

Run python ML models in a service

## installation

### extras

#### fastapi-uvicorn

Run ML models as HTTP service using fastapi & uvicorn

After the service is started, you may find and try all the handlers at `/docs` path.

## usage

### example

[examples/myproject/models.py](src/examples/myproject/models.py)

* run dynamic models (each model can be trained by provided input)
  ```
  $ ml2service \
      examples.myproject.models:FooDynamicModelTrainer \
      dynamic \
      http

  ...
  INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
  ```
  ![fastapi docs at http://127.0.0.1:8000/docs](docs/examples/foo-dynamic-docs.png)
* run static model (one model will be trained at the start)
  ```
  $ ml2service \
      examples.myproject.models:FooStaticModelTrainer \
      static examples/myproject/data.json \
      http
  
  ...
  INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
  ```
  ![fastapi docs at http://127.0.0.1:8000/docs](docs/examples/foo-static-docs.png)
