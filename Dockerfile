FROM python:3.11-slim

WORKDIR /app

# 国内网络：pip 走清华镜像 + setuptools 预装并关闭 build isolation——
# 否则 pip 会开隔离子进程去 PyPI 拉 setuptools，直连一断就 build 失败
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "setuptools>=68" \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY pyproject.toml ./
COPY vram_calc ./vram_calc
RUN pip install --no-cache-dir --no-build-isolation .

EXPOSE 8000

# bundled data ships inside the package; user-added models/gpus land in
# /data (mounted volume) via VRAM_CALC_HOME, surviving container rebuilds
ENV VRAM_CALC_HOME=/data

CMD ["uvicorn", "vram_calc.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
