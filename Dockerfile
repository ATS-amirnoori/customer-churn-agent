# Use a lightweight Python 3.12 environment as the application's base.
FROM python:3.12-slim


# Store the application files inside /app in the container.
WORKDIR /app


# Copy the dependency list first so Docker can install the Python packages.
COPY requirements.txt .


# Install all application dependencies.
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the project into the container.
COPY . .


# Document the port used by Streamlit.
EXPOSE 8501


# Start the Streamlit application when the container launches.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]