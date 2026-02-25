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
LLM-based discovery of figure briefs from a paper document.
"""

from __future__ import annotations

from typing import Dict, List

import json_repair
from google.genai import types

from . import generation_utils
from .paper_sections import (
    extract_sections,
    rank_sections_for_figure_discovery,
    build_discovery_context,
)


DISCOVERY_SYSTEM_PROMPT = """
You are an expert research assistant helping authors identify high-value academic diagrams.
Given paper content, propose candidate figure briefs for conceptual/method diagrams.

Rules:
1. Focus on methods, architectures, data flow, pipelines, training/evaluation workflows.
2. Each brief should be specific and useful for a standalone figure.
3. Keep captions concise and paper-like.
4. Do not invent claims not supported by the paper text.
5. Return strictly valid JSON matching the required schema.
"""


def _sanitize_briefs(raw_obj: object, max_figures: int) -> List[Dict[str, str]]:
    if not isinstance(raw_obj, dict):
        return []

    briefs = raw_obj.get("figure_briefs", [])
    if not isinstance(briefs, list):
        return []

    cleaned: List[Dict[str, str]] = []
    for idx, item in enumerate(briefs[:max_figures]):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        source_section = str(item.get("source_section_title", "")).strip()
        source_excerpt = str(item.get("source_excerpt", "")).strip()
        caption_draft = str(item.get("caption_draft", "")).strip()
        rationale = str(item.get("rationale", "")).strip()

        if not title or not source_excerpt or not caption_draft:
            continue

        cleaned.append(
            {
                "brief_id": f"brief_{idx + 1}",
                "title": title,
                "source_section_title": source_section or "Unknown Section",
                "source_excerpt": source_excerpt,
                "caption_draft": caption_draft,
                "rationale": rationale,
            }
        )
    return cleaned


async def discover_figure_briefs(
    markdown_text: str,
    paper_title: str,
    model_name: str,
    max_figures: int = 8,
    max_sections: int = 8,
) -> Dict[str, object]:
    """
    Discover candidate figure briefs from paper markdown text.
    """
    sections = extract_sections(markdown_text)
    ranked_sections = rank_sections_for_figure_discovery(sections, max_sections=max_sections)
    context_text = build_discovery_context(ranked_sections, max_chars=18000)

    if not context_text:
        return {
            "briefs": [],
            "sections_analyzed": 0,
            "raw_response": "",
        }

    user_prompt = f"""
Paper title: {paper_title}

Analyze the paper sections below and propose up to {max_figures} candidate method-diagram figure briefs.

Return JSON with this exact structure:
{{
  "figure_briefs": [
    {{
      "title": "short figure title",
      "source_section_title": "section name",
      "source_excerpt": "copy a short excerpt supporting this figure",
      "caption_draft": "Figure X: ...",
      "rationale": "why this figure helps the paper"
    }}
  ]
}}

Paper sections:
{context_text}
"""

    response_list = await generation_utils.call_gemini_with_retry_async(
        model_name=model_name,
        contents=[{"type": "text", "text": user_prompt}],
        config=types.GenerateContentConfig(
            system_instruction=DISCOVERY_SYSTEM_PROMPT,
            temperature=0.2,
            candidate_count=1,
            max_output_tokens=12000,
        ),
        max_attempts=5,
        retry_delay=5,
        error_context="figure brief discovery",
    )

    raw_response = response_list[0].strip() if response_list else ""
    clean_json = raw_response.replace("```json", "").replace("```", "").strip()

    parsed_obj: object
    try:
        parsed_obj = json_repair.loads(clean_json)
    except Exception:
        parsed_obj = {}

    briefs = _sanitize_briefs(parsed_obj, max_figures=max_figures)

    return {
        "briefs": briefs,
        "sections_analyzed": len(ranked_sections),
        "raw_response": raw_response,
    }

