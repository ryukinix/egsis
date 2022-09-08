FROM python:3.10

# for scipy
RUN apt update && apt install gfortran libatlas-base-dev -y
RUN pip install pdm setuptools wheel

WORKDIR /app
COPY pyproject.toml pdm.lock /app/
RUN pdm install --no-self

COPY egsis README.md setup.py egsis /app/
RUN pdm install

CMD ["pdm", "run", "egsis"]
ENTRYPOINT ["pdm", "run"]
