#docker build -t tab-survey .
docker run -v ~/output:/opt/notebooks/output -p 3123:3123 --rm -it tab-survey
#docker run -v ~/output:/opt/notebooks/output -p 3123:3123 --rm -it --gpus all tab-survey
# opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=3123 --no-browser --allow-root