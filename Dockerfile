# Stage 1: build the single-page app.
FROM node:22-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# Stage 2: the shared base for the two PDF builds.
#
# The PDFs are built in the image so that a deploy is only a pull: the image
# carries the book and slides that match its own chapters, and no workstation
# has to build and copy them. Chromium, fonts, node_modules, and the loaders'
# Python packages stay in these stages. The final image receives only the PDFs
# and their manifests.
#
# These stages use Debian rather than Alpine because Chromium and its fonts are
# better supported there, and because the print layouts need a node_modules
# built for glibc, which the Alpine `frontend` stage can't provide.
#
# Layers run from the least to the most frequently changed input. Don't pass
# GIT_SHA into these stages: a per-commit value would rebuild both PDFs on every
# push. The manifests' commit is "unknown" for that reason.
FROM node:22-bookworm-slim AS pdf-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        chromium python3 fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV CHROME_PATH=/usr/bin/chromium \
    PYTHON=/opt/venv/bin/python

WORKDIR /pdf

COPY frontend/package.json frontend/package-lock.json* ./frontend/
RUN npm --prefix frontend ci --no-audit --no-fund

# uv resolves and installs several times faster than pip, and its venv needs
# no python3-venv package.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
COPY backend/requirements.txt ./backend/requirements.txt
RUN uv venv --python /usr/bin/python3 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r backend/requirements.txt

# Stage 3: print the PDF book. The two PDF stages build in parallel and cache
# independently, so a change to a slide deck doesn't reprint the book.
FROM pdf-base AS book

# Only what the book is made from, as build-book.mjs hashes it: the print
# layout's import closure, not all of src/, so a change to the site's UI or to
# a deck doesn't reprint the book. Keep this list in step with BOOK_SOURCES in
# build-book.mjs. A file missing here fails the Vite build with an unresolved
# import. PRINT_ENTRY=book makes Vite build book.html alone, so slides.html and
# the slides layout stay out.
COPY backend/app/__init__.py backend/app/config.py backend/app/curriculum.py ./backend/app/
COPY scripts/dump_book.py ./scripts/dump_book.py
COPY content/ ./content/
COPY frontend/book.html frontend/vite.book.config.ts frontend/tsconfig.json ./frontend/
COPY frontend/public/ ./frontend/public/
COPY frontend/src/book/ ./frontend/src/book/
COPY frontend/src/ui/*.css ./frontend/src/ui/
COPY frontend/src/index.css ./frontend/src/
COPY frontend/src/components/Brand.tsx frontend/src/components/Markdown.tsx frontend/src/components/MermaidBlock.tsx ./frontend/src/components/
COPY frontend/src/components/viz/ ./frontend/src/components/viz/
COPY frontend/src/lib/chapters.ts frontend/src/lib/katexMacros.json frontend/src/lib/markdownPlugins.ts ./frontend/src/lib/
COPY frontend/scripts/build-book.mjs ./frontend/scripts/
RUN node frontend/scripts/build-book.mjs --force --out /out \
    && rm -rf /out/.build

# Stage 4: print the slide decks, from the inputs build-slides.mjs hashes. The
# list is the import closure of src/slides/main.tsx. Keep it in step with
# SLIDES_SOURCES in build-slides.mjs.
FROM pdf-base AS slides

COPY backend/app/__init__.py backend/app/config.py backend/app/curriculum.py backend/app/decks.py ./backend/app/
COPY scripts/dump_slides.py ./scripts/dump_slides.py
COPY content/chapters/ ./content/chapters/
COPY content/slides/ ./content/slides/
COPY frontend/slides.html frontend/vite.book.config.ts frontend/tsconfig.json ./frontend/
COPY frontend/public/ ./frontend/public/
COPY frontend/src/slides/ ./frontend/src/slides/
COPY frontend/src/ui/*.css ./frontend/src/ui/
COPY frontend/src/index.css ./frontend/src/
COPY frontend/src/components/DeckPrint.tsx frontend/src/components/Markdown.tsx frontend/src/components/MermaidBlock.tsx frontend/src/components/SlideFrame.tsx ./frontend/src/components/
COPY frontend/src/components/viz/ ./frontend/src/components/viz/
COPY frontend/src/lib/katexMacros.json frontend/src/lib/markdownPlugins.ts ./frontend/src/lib/
COPY frontend/scripts/build-slides.mjs ./frontend/scripts/
RUN node frontend/scripts/build-slides.mjs --force --out /out \
    && rm -rf /out/.slides-build

# Stage 5: the Python service, which also serves the built assets.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1 \
    DATA_DIR=/data

WORKDIR /app

# uv installs into the system Python, as pip did, several times faster.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

COPY backend/requirements.txt ./backend/requirements.txt
RUN uv pip install -r backend/requirements.txt

COPY backend/ ./backend/
COPY content/ ./content/
COPY gpu/ ./gpu/
# The MCP server's get_document tool reads these.
COPY docs/ ./docs/
COPY --from=frontend /build/dist ./frontend/dist

# The commit this image was built from, reported by /api/health. Declared after
# the COPY steps so changing it does not invalidate the layers above.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown -R app:app /data /app

# After the chown above, with its own --chown, so the PDFs are stored once in
# the image and not a second time as a chown copy.
COPY --from=book --chown=app:app /out/gpu-programming-for-ml.pdf /out/manifest.json ./book/
COPY --from=slides --chown=app:app /out/gpu-programming-for-ml-slides.pdf /out/slides-manifest.json ./book/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
