FROM python:3.11-slim

WORKDIR /app

# deps first (cached layer) -- no build isolation needed in a clean container
COPY pyproject.toml ./
COPY vram_calc ./vram_calc
RUN pip install --no-cache-dir .

EXPOSE 8000

# bundled data ships inside the package; user-added models/gpus land in
# /data (mounted volume) via VRAM_CALC_HOME, surviving container rebuilds
ENV VRAM_CALC_HOME=/data

CMD ["uvicorn", "vram_calc.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
