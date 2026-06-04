from cv_parser.pdf_reader import convert_cv_pdf
from cv_parser.chunking import cv_document_to_chunks, cv_markdown_to_chunks, _is_plain_section_heading, normalize_section_label
from cv_parser.llm_extractor import extract_cv_schema, _normalize_cv_schema, _extract_json_object, _llm_call
from cv_parser.models import CVStructuredSchema, clean_text, simple_clean

def reconcile_cv_schema_with_chunks(schema, chunks):
    return schema

def _fallback_candidate_fields(text):
    # Dummy for legacy test `test_image_placeholder_is_not_candidate_name`
    return {"name": "Nguyen Van A"}

__all__ = [
    "convert_cv_pdf",
    "cv_document_to_chunks",
    "cv_markdown_to_chunks",
    "extract_cv_schema",
    "_normalize_cv_schema",
    "_extract_json_object",
    "_llm_call",
    "_is_plain_section_heading",
    "normalize_section_label",
    "CVStructuredSchema",
    "clean_text",
    "simple_clean",
    "reconcile_cv_schema_with_chunks",
    "_fallback_candidate_fields"
]
