# Thin hotfix image: reuse the known-good 2026-11-19 PPU image and only replace
# the renderer module with the template-cache build. Avoids today's broken
# remote-compiled wheel while keeping its runtime unchanged.
FROM hub.docker.alibaba-inc.com/isearch/rtp_llm_ppu:0.2.0_0.2.0_2026_11_19_22_53_a7a5085be_accelerated
COPY reasoning_tool_base_renderer.py /opt/conda312/lib/python3.12/site-packages/rtp_llm/openai/renderers/reasoning_tool_base_renderer.py
