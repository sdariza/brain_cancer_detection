FROM continuumio/anaconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "deploy_model", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "deploy_model", "python", "app.py"]
