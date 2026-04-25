# =============================================================================
# SEMANTIC CHUNKING ENGINE
# =============================================================================
# Sử dụng embedding vectors để:
# 1. Tìm các từ/heading có cùng ngữ nghĩa
# 2. Normalize sections dựa trên semantic similarity thay vì keyword
# 3. Merge chunks dựa trên semantic relevance
# =============================================================================

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
from dotenv import load_dotenv

# Import embedding functions
from bedrock_utils import get_embedding as bedrock_get_embedding

load_dotenv()


# =============================================================================
# 1. EMBEDDING ENGINE (với caching)
# =============================================================================

@dataclass
class EmbeddingCache:
    """Cache cho embeddings để tránh call API liên tục."""
    cache: Dict[str, List[float]] = None
    
    def __post_init__(self):
        if self.cache is None:
            self.cache = {}
    
    def get(self, text: str) -> Optional[List[float]]:
        return self.cache.get(text)
    
    def set(self, text: str, embedding: List[float]):
        self.cache[text] = embedding
    
    def save(self, filepath: str):
        """Lưu cache ra file để tái sử dụng."""
        with open(filepath, 'w') as f:
            json.dump(self.cache, f)
    
    def load(self, filepath: str):
        """Tải cache từ file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.cache = json.load(f)


_embedding_cache = EmbeddingCache()


def get_embedding(text: str, use_cache: bool = True) -> List[float]:
    """
    Lấy embedding cho text từ AWS Bedrock Titan.
    
    Args:
        text: Text cần embedding
        use_cache: Có dùng cache không (tránh gọi API liên tục)
    
    Returns:
        List[float]: Embedding vector (1536 dims cho Titan)
    """
    text = text.strip()
    if not text:
        return []
    
    # Check cache
    if use_cache:
        cached = _embedding_cache.get(text)
        if cached is not None:
            return cached
    
    try:
        embedding = bedrock_get_embedding(text)
        if embedding:
            _embedding_cache.set(text, embedding)
            return embedding
    except Exception as e:
        print(f"[ERROR] Bedrock embedding failed: {e}")
        raise
    
    # Fallback: trả về zero vector
    print(f"[WARNING] Không thể tạo embedding cho: {text[:50]}... Sử dụng zero vector")
    return [0.0] * 1536


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Tính cosine similarity giữa 2 vectors."""
    if not vec1 or not vec2:
        return 0.0
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# =============================================================================
# 2. STANDARD SECTION DEFINITIONS
# =============================================================================

STANDARD_SECTIONS = {
    "experience": {
        "keywords": ["work experience", "professional experience", "employment", "internship", "kinh nghiem"],
        "description": "Kinh nghiệm làm việc, dự án từng tham gia"
    },
    "education": {
        "keywords": ["education", "academic", "degree", "university", "college", "hoc van"],
        "description": "Học vấn, bằng cấp, trường học"
    },
    "skills": {
        "keywords": ["skills", "technical", "competencies", "expertise", "ky nang"],
        "description": "Kỹ năng kỹ thuật, năng lực"
    },
    "summary": {
        "keywords": ["summary", "objective", "profile", "about", "tom tat"],
        "description": "Tóm tắt, mục tiêu nghề nghiệp"
    },
    "certifications": {
        "keywords": ["certificate", "certification", "license", "chung chi"],
        "description": "Chứng chỉ, giấy phép, bằng cấp chuyên môn"
    },
    "activities": {
        "keywords": ["activity", "volunteer", "research", "extracurricular", "hoat dong"],
        "description": "Hoạt động xã hội, tình nguyện, nghiên cứu"
    },
    "projects": {
        "keywords": ["project", "portfolio", "du an"],
        "description": "Dự án cá nhân, portfolio"
    },
    "awards": {
        "keywords": ["award", "honor", "achievement", "giai thuong"],
        "description": "Giải thưởng, thành tích"
    },
    "contact": {
        "keywords": ["contact", "information", "phone", "email", "language", "lien he"],
        "description": "Thông tin liên hệ, ngôn ngữ"
    }
}


# Cache embeddings cho standard sections
_standard_embeddings_cache = {}


def _get_standard_section_embeddings() -> Dict[str, Tuple[List[float], List[List[float]]]]:
    """
    Tính embeddings cho tất cả standard sections một lần (caching).
    
    Returns:
        Dict[section_name -> (section_embedding, keyword_embeddings)]
    """
    global _standard_embeddings_cache
    
    if _standard_embeddings_cache:
        return _standard_embeddings_cache
    
    print("[INFO] Tính embeddings cho standard sections...")
    
    for section_name, section_info in STANDARD_SECTIONS.items():
        # Embedding cho description (representative của section)
        description = section_info['description']
        section_embedding = get_embedding(description)
        
        # Embeddings cho từng keyword
        keyword_embeddings = [get_embedding(kw) for kw in section_info['keywords']]
        
        _standard_embeddings_cache[section_name] = (section_embedding, keyword_embeddings)
    
    print(f"[INFO] Đã tính xong embeddings cho {len(_standard_embeddings_cache)} sections")
    return _standard_embeddings_cache


# =============================================================================
# 3. SEMANTIC SECTION DETECTION
# =============================================================================

def find_best_section_semantic(
    text: str,
    threshold: float = 0.5,
    top_k: int = 1
) -> Tuple[str, float]:
    """
    Tìm best-matching section dựa trên semantic similarity với embeddings.
    Thay vì keyword matching, sử dụng vector similarity.
    
    Args:
        text: Heading/text cần classify
        threshold: Minimum similarity score
        top_k: Số kết quả tốt nhất trả về
    
    Returns:
        (section_name, similarity_score)
    """
    text = text.strip().lower()
    if not text:
        return "other", 0.0
    
    # Lấy embedding cho input text
    text_embedding = get_embedding(text)
    if not text_embedding:
        return "other", 0.0
    
    # So sánh với embeddings của tất cả standard sections
    section_embeddings = _get_standard_section_embeddings()
    
    scores: List[Tuple[str, float]] = []
    
    for section_name, (section_emb, keyword_embs) in section_embeddings.items():
        # Tính similarity với section description
        section_sim = cosine_similarity(text_embedding, section_emb)
        
        # Tính max similarity với bất kỳ keyword nào
        keyword_sims = [cosine_similarity(text_embedding, kw_emb) for kw_emb in keyword_embs]
        keyword_sim = max(keyword_sims) if keyword_sims else 0.0
        
        # Combine: prioritize description, secondary use max keyword
        combined_score = 0.6 * section_sim + 0.4 * keyword_sim
        scores.append((section_name, combined_score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top match nếu vượt threshold
    best_section, best_score = scores[0]
    if best_score >= threshold:
        return best_section, best_score
    else:
        return "other", best_score


def find_best_sections_semantic(
    text: str,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Trả về top-k matching sections có scores.
    Hữu ích để debug hoặc visualization.
    """
    text = text.strip().lower()
    if not text:
        return [("other", 0.0)]
    
    text_embedding = get_embedding(text)
    if not text_embedding:
        return [("other", 0.0)]
    
    section_embeddings = _get_standard_section_embeddings()
    scores: List[Tuple[str, float]] = []
    
    for section_name, (section_emb, keyword_embs) in section_embeddings.items():
        section_sim = cosine_similarity(text_embedding, section_emb)
        keyword_sims = [cosine_similarity(text_embedding, kw_emb) for kw_emb in keyword_embs]
        keyword_sim = max(keyword_sims) if keyword_sims else 0.0
        combined_score = 0.6 * section_sim + 0.4 * keyword_sim
        scores.append((section_name, combined_score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# =============================================================================
# 4. SEMANTIC HEADING NORMALIZATION
# =============================================================================

def normalize_headings_semantic(
    headings: List[str],
    use_semantic: bool = True
) -> str:
    """
    Normalize heading path thành một section label duy nhất.
    
    Nếu use_semantic=True:
        - Dùng semantic matching để tìm best-match section
        - Tính score dựa trên tất cả headings
    
    Nếu use_semantic=False:
        - Fallback to keyword (legacy behavior)
    """
    if not headings:
        return "other"
    
    if not use_semantic:
        # Legacy: dùng last heading keyword matching
        from pdf_utils import _heading_to_section, fix_spaced_letters
        headings_fixed = [fix_spaced_letters(h).strip() for h in headings]
        return _heading_to_section(headings_fixed)
    
    # Semantic approach: combine insights từ tất cả headings
    # Ưu tiên heading cụ thể nhất (cuối cùng)
    combined_scores: Dict[str, float] = {}
    
    for i, heading in enumerate(reversed(headings)):
        # Weighted by position (later headings have higher weight)
        weight = 1.0 + (i * 0.5)
        
        results = find_best_sections_semantic(heading, top_k=3)
        for section_name, score in results:
            if section_name not in combined_scores:
                combined_scores[section_name] = 0.0
            combined_scores[section_name] += score * weight
    
    # Normalize scores
    total_score = sum(combined_scores.values())
    if total_score > 0:
        combined_scores = {
            section: score / total_score
            for section, score in combined_scores.items()
        }
    
    # Return best section
    if combined_scores:
        best_section = max(combined_scores.items(), key=lambda x: x[1])
        return best_section[0]
    
    return "other"


# =============================================================================
# 5. SEMANTIC CHUNK MERGING
# =============================================================================

def should_merge_chunks_semantic(
    chunk1: Dict,
    chunk2: Dict,
    similarity_threshold: float = 0.7
) -> bool:
    """
    Quyết định có nên merge 2 chunks dựa trên:
    1. Cùng section
    2. Semantic similarity cao giữa chunk texts
    """
    # Rule 1: Phải cùng section
    if chunk1.get("section") != chunk2.get("section"):
        return False
    
    # Rule 2: Cùng heading path
    if chunk1.get("headings") == chunk2.get("headings"):
        return True
    
    # Rule 3: Semantic similarity cao
    text1 = chunk1.get("text", "").strip()
    text2 = chunk2.get("text", "").strip()
    
    if len(text1) < 20 or len(text2) < 20:
        # Text quá ngắn, không đủ semantic signal
        return chunk1.get("headings") == chunk2.get("headings")
    
    emb1 = get_embedding(text1[:500])  # Limit to first 500 chars
    emb2 = get_embedding(text2[:500])
    
    similarity = cosine_similarity(emb1, emb2)
    return similarity >= similarity_threshold


def merge_semantically_similar_chunks(
    chunks: List[Dict],
    min_chars: int = 80,
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """
    Merge chunks dựa trên:
    1. Cùng section
    2. Semantic similarity cao
    3. Size nhỏ hơn min_chars
    """
    if not chunks:
        return chunks
    
    merged = [chunks[0].copy()]
    
    for curr in chunks[1:]:
        prev = merged[-1]
        curr_size = len(curr.get("text", ""))
        
        # Only merge nếu current chunk quá nhỏ
        if curr_size < min_chars and should_merge_chunks_semantic(
            prev, curr, similarity_threshold
        ):
            # Merge texts
            prev["text"] = prev["text"] + "\n" + curr["text"]
        else:
            merged.append(curr.copy())
    
    return merged


# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================

def debug_heading_scores(heading: str) -> None:
    """In ra top-5 matching sections với scores để debug."""
    print(f"\nDEBUG Heading: '{heading}'")
    print("─" * 60)
    
    results = find_best_sections_semantic(heading, top_k=5)
    for section_name, score in results:
        print(f"  {section_name:20s} | Similarity: {score:.4f}")


def save_embedding_cache(filepath: str = ".embedding_cache.json") -> None:
    """Lưu embedding cache để tái sử dụng."""
    _embedding_cache.save(filepath)
    print(f"[INFO] Đã lưu embedding cache: {filepath}")


def load_embedding_cache(filepath: str = ".embedding_cache.json") -> None:
    """Tải embedding cache từ file."""
    _embedding_cache.load(filepath)
    print(f"[INFO] Đã tải embedding cache: {filepath}")


# =============================================================================
# 7. CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Chunking Engine")
    parser.add_argument("--test-heading", type=str, help="Test semantic section detection")
    parser.add_argument("--test-all", action="store_true", help="Test tất cả standard sections")
    parser.add_argument("--save-cache", action="store_true", help="Lưu embedding cache")
    
    args = parser.parse_args()
    
    if args.test_all:
        print("\n" + "="*60)
        print("Testing Semantic Section Detection")
        print("="*60)
        
        test_headings = [
            "Work Experience",
            "Công việc từng làm",
            "Kinh nghiệm chuyên môn",
            "Education & Degree",
            "Học vấn",
            "Technical Skills",
            "Kỹ năng",
            "Summary",
            "Tóm tắt CV",
            "Certifications",
            "Chứng chỉ",
            "Awards & Achievements",
            "Giải thưởng",
        ]
        
        for heading in test_headings:
            debug_heading_scores(heading)
    
    elif args.test_heading:
        debug_heading_scores(args.test_heading)
    
    elif args.save_cache:
        save_embedding_cache()