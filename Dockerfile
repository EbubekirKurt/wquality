FROM python:3.9

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

USER root
RUN apt-get update && apt-get install -y git-lfs
USER user

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN git lfs install
RUN git lfs pull

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
