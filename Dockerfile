# QSAID Work Instructions API — backend only (no UI)
FROM python:3.11-slim-bookworm

WORKDIR /app

# OpenCV runtime libraries (opencv-python)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (pipeline modules + API package)
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY pipeline.py audio_transcription.py azure_client.py key_frame_generator.py \
     report_builder.py sop_stateless.py time_utils.py vision_analysis.py ./

# Windows editors may save CRLF; Linux container needs LF for shell scripts
RUN sed -i 's/\r$//' scripts/startup.sh && chmod +x scripts/startup.sh
ENV API_PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Azure App Service / Container Apps set PORT; local/docker default 8000
CMD ["sh", "scripts/startup.sh"]
