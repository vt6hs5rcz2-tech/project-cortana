"""Shared helpers for Knowledge Vault document tests."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfWriter

# Hand-built minimal text PDF used by extraction/ingestion tests.
# PDF xref entries require a trailing space before the EOL marker; keep those
# spaces inside concatenated literals (not as end-of-line whitespace) so
# EditorConfig trim_trailing_whitespace cannot silently corrupt the fixture.
TEXT_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
    b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
    b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
    b"4 0 obj<< /Length 55 >>stream\n"
    b"BT /F1 24 Tf 100 700 Td (Hello PDF) Tj ET\n"
    b"endstream\n"
    b"endobj\n"
    b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000371 00000 n \n"
    b"trailer<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n"
    b"448\n"
    b"%%EOF"
)


def blank_pdf_bytes() -> bytes:
    """Return a PDF that contains no extractable text."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def encrypted_pdf_bytes() -> bytes:
    """Return an encrypted PDF that cannot be ingested in this milestone."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret-password")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
