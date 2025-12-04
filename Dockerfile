FROM python:3.9

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Install requirements and git-lfs
USER root
RUN apt-get update && apt-get install -y git-lfs
USER user

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Fetch LFS files (if they are pointers)
RUN git lfs install
RUN git lfs pull

# Expose port 7860
EXPOSE 7860

# Run the application
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
