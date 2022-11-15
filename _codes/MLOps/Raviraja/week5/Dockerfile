FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements_inference.txt
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "Week5:app", "--host", "0.0.0.0", "--port", "8000"]