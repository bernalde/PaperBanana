# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helpers to extract ranked sections from markdown-like paper text.
"""

from __future__ import annotations

import re
from typing import Dict, List


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

SECTION_KEYWORDS = [
    "method",
    "methodology",
    "approach",
    "framework",
    "architecture",
    "pipeline",
    "system",
    "algorithm",
    "training",
    "implementation",
]


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def extract_sections(markdown_text: str) -> List[Dict[str, object]]:
    """
    Parse markdown headings and return sections with content.
    """
    lines = markdown_text.splitlines()
    sections: List[Dict[str, object]] = []

    current_title = "Document"
    current_level = 1
    current_lines: List[str] = []

    def flush_current() -> None:
        nonlocal current_title, current_level, current_lines
        body = "\n".join(current_lines).strip()
        if body:
            section_id = f"sec_{len(sections) + 1}"
            sections.append(
                {
                    "section_id": section_id,
                    "title": current_title,
                    "level": current_level,
                    "content": body,
                    "word_count": _count_words(body),
                }
            )

    for line in lines:
        match = HEADING_PATTERN.match(line)
        if match:
            flush_current()
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)

    flush_current()

    if not sections:
        cleaned = markdown_text.strip()
        if cleaned:
            sections = [
                {
                    "section_id": "sec_1",
                    "title": "Document",
                    "level": 1,
                    "content": cleaned,
                    "word_count": _count_words(cleaned),
                }
            ]
    return sections


def _section_score(section: Dict[str, object]) -> float:
    title = str(section.get("title", "")).lower()
    content = str(section.get("content", "")).lower()
    word_count = int(section.get("word_count", 0))

    score = 0.0
    for keyword in SECTION_KEYWORDS:
        if keyword in title:
            score += 5.0
        score += content.count(keyword) * 0.5

    # Prefer medium-length sections over very short or excessively long ones.
    if 120 <= word_count <= 1200:
        score += 3.0
    elif 60 <= word_count < 120:
        score += 1.5
    elif word_count > 1200:
        score -= 0.5

    return score


def rank_sections_for_figure_discovery(
    sections: List[Dict[str, object]],
    max_sections: int = 8,
) -> List[Dict[str, object]]:
    """
    Rank sections likely to contain figure-worthy method details.
    """
    scored_sections = []
    for section in sections:
        section_copy = dict(section)
        section_copy["score"] = _section_score(section_copy)
        scored_sections.append(section_copy)

    scored_sections.sort(key=lambda item: (item["score"], item.get("word_count", 0)), reverse=True)
    return scored_sections[:max_sections]


def build_discovery_context(
    ranked_sections: List[Dict[str, object]],
    max_chars: int = 18000,
) -> str:
    """
    Concatenate ranked sections into a bounded prompt context.
    """
    chunks: List[str] = []
    current_len = 0

    for section in ranked_sections:
        block = (
            f"[Section: {section['title']}]\n"
            f"{section['content']}\n"
        )
        if current_len + len(block) > max_chars and chunks:
            break
        chunks.append(block)
        current_len += len(block)

    return "\n".join(chunks).strip()

