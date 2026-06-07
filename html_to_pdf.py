#!/usr/bin/env python3
"""Render the already-built report.html to a static PDF using Chromium (Playwright).

OJS / D3 visualisations render client-side, so we load the page, wait for the
network to go idle, scroll through the whole document to trigger any lazy chart
rendering, then wait until the number of rendered <svg>/<canvas> elements is
stable before printing to PDF.
"""
import io
import os
import threading
import functools
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "report.html")
PDF = os.path.join(HERE, "report.pdf")
VECTOR_PDF = os.path.join(HERE, "_report_vector.pdf")

# The charts are vector-heavy (choropleth map, heatmaps), so the raw PDF is large.
# We rasterise pages to JPEG to keep the file comfortably under this limit.
MAX_MB = 10.0


def start_server():
    """Serve the report directory over HTTP so OJS will run (it refuses on file://)."""
    handler = functools.partial(SimpleHTTPRequestHandler, directory=HERE)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


def compress_pdf(src_path, out_path, max_mb=MAX_MB):
    """Rasterise pages to JPEG and rebuild, stepping DPI down until under max_mb."""
    import fitz
    from PIL import Image

    for dpi, q in [(200, 82), (170, 80), (150, 78), (130, 75), (110, 72)]:
        src = fitz.open(src_path)
        doc = fitz.open()
        for p in src:
            pix = p.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            buf.seek(0)
            r = p.rect
            page = doc.new_page(width=r.width, height=r.height)
            page.insert_image(r, stream=buf.read())
            # Rasterising drops the hyperlink annotations, so copy them back.
            # Page dimensions match the source, so link rectangles align exactly.
            for link in p.get_links():
                if link.get("uri"):
                    page.insert_link(link)
        doc.save(out_path, deflate=True)
        doc.close()
        src.close()
        size = os.path.getsize(out_path) / 1e6
        print(f"  raster dpi={dpi} q={q} -> {size:.2f} MB")
        if size <= max_mb:
            return size
    return os.path.getsize(out_path) / 1e6


def count_graphics(page):
    return page.evaluate(
        "() => document.querySelectorAll('svg, canvas, .plot-container').length"
    )


def main():
    httpd, port = start_server()
    url = f"http://127.0.0.1:{port}/report.html"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1600})
        print("loading", url)
        page.goto(url, wait_until="networkidle", timeout=120000)

        # Give OJS its initial run (it loads data and builds charts asynchronously).
        page.wait_for_timeout(9000)

        # Scroll through the document in steps so every cell enters the viewport
        # and any intersection-observer-driven rendering kicks in.
        total = page.evaluate("() => document.body.scrollHeight")
        step = 1200
        y = 0
        while y < total:
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(350)
            y += step
            total = page.evaluate("() => document.body.scrollHeight")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(3000)
        page.evaluate("window.scrollTo(0, 0)")

        # Wait for the graphics count to stabilise.
        prev = -1
        for _ in range(20):
            cur = count_graphics(page)
            print("graphics elements:", cur)
            if cur == prev and cur > 0:
                break
            prev = cur
            page.wait_for_timeout(1500)

        page.wait_for_timeout(2000)

        # Chromium's paginated page.pdf() path renders box-shadows / backdrop-filter
        # and translucent card backgrounds as offset solid-grey "ghost" boxes.
        # These are purely decorative, so we strip them for the static PDF only.
        page.add_style_tag(content="""
          * {
            box-shadow: none !important;
            -webkit-box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            text-shadow: none !important;
          }
          .pipeline-diagram { background: #ffffff !important; }
          .pipeline-step { background: #ffffff !important; }
        """)
        page.wait_for_timeout(400)

        page.pdf(
            path=VECTOR_PDF,
            print_background=True,
            prefer_css_page_size=False,
            format="A4",
            margin={"top": "12mm", "bottom": "14mm", "left": "10mm", "right": "10mm"},
            scale=0.72,
        )
        browser.close()
    httpd.shutdown()

    print(f"vector PDF: {os.path.getsize(VECTOR_PDF)/1e6:.2f} MB; compressing to <= {MAX_MB} MB")
    final = compress_pdf(VECTOR_PDF, PDF)
    os.remove(VECTOR_PDF)
    print(f"wrote {PDF} ({final:.2f} MB)")


if __name__ == "__main__":
    main()
