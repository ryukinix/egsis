FROM ryukinix/pdm:3.10

COPY pyproject.toml pdm.lock /app/
RUN pdm install --no-self

ADD README.md setup.py /app/
COPY egsis /app/egsis
RUN pdm install --no-editable
CMD ["pdm", "run", "egsis"]
