# Thin hotfix image: reuse the known-good PPU image and replace only the Python
# frontend files needed for the timing metrics. Build context is the repo root:
#   docker build -f docker/ppu_hotfix/Dockerfile -t <tag> .
FROM hub.docker.alibaba-inc.com/isearch/rtp_llm_ppu:0.2.0_0.2.0_2026_11_24_12_31_dc51a50f5_accelerated
COPY rtp_llm/openai/renderers/reasoning_tool_base_renderer.py /opt/conda312/lib/python3.12/site-packages/rtp_llm/openai/renderers/reasoning_tool_base_renderer.py
COPY rtp_llm/frontend/frontend_server.py /opt/conda312/lib/python3.12/site-packages/rtp_llm/frontend/frontend_server.py
COPY rtp_llm/frontend/frontend_app.py /opt/conda312/lib/python3.12/site-packages/rtp_llm/frontend/frontend_app.py
COPY rtp_llm/metrics/kmonitor_metric_reporter.py /opt/conda312/lib/python3.12/site-packages/rtp_llm/metrics/kmonitor_metric_reporter.py
RUN python3 -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8'), filename=p) for p in ('/opt/conda312/lib/python3.12/site-packages/rtp_llm/openai/renderers/reasoning_tool_base_renderer.py', '/opt/conda312/lib/python3.12/site-packages/rtp_llm/frontend/frontend_server.py', '/opt/conda312/lib/python3.12/site-packages/rtp_llm/frontend/frontend_app.py', '/opt/conda312/lib/python3.12/site-packages/rtp_llm/metrics/kmonitor_metric_reporter.py')]"
