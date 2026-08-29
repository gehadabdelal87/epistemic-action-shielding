FROM python:3.13-slim
WORKDIR /opt/eas
COPY . /opt/eas
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e '.[experiments,test]'
CMD ["python", "-m", "pytest"]
