FROM ryukinix/pdm:3.11.3

COPY pyproject.toml setup.cfg pdm.lock /app/
RUN pdm install --no-self

ADD README.md setup.py /app/
COPY egsis /app/egsis
RUN pdm install --no-editable

RUN chmod -R 777 /app /tmp
CMD ["pdm", "run", "egsis"]
