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
Parallel Streamlit Demo for PaperVizAgent
Accepts user text input, duplicates it 10 times, and runs parallel processing
to generate multiple diagram candidates for comparison.
"""

import streamlit as st
import asyncio
import base64
import json
import re
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("DEBUG: Importing agents...")
try:
    from agents.planner_agent import PlannerAgent
    print("DEBUG: Imported PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("DEBUG: Imported all agents")
    from utils import config
    from utils.paperviz_processor import PaperVizProcessor
    from utils.paper_ingest import convert_document_to_markdown
    from utils.figure_discovery import discover_figure_briefs
    from utils.paper_sections import extract_sections, rank_sections_for_figure_discovery
    print("DEBUG: Imported utils")

    import yaml
    config_path = Path(__file__).parent / "configs" / "model_config.yaml"
    model_config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config_data = yaml.safe_load(f) or {}

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

except ImportError as e:
    print(f"DEBUG: ImportError: {e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"DEBUG: Exception during import: {e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperVizAgent Parallel Demo",
    page_icon="🍌"
)

def clean_text(text):
    """Clean text by removing invalid UTF-8 surrogate characters."""
    if not text:
        return text
    if isinstance(text, str):
        # Remove surrogate characters that cause UnicodeEncodeError
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """Convert base64 string to PIL Image."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def create_sample_inputs(method_content, caption, diagram_type="Pipeline", aspect_ratio="16:9", num_copies=10, max_critic_rounds=3):
    """Create multiple copies of the input data for parallel processing."""
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": method_content,
        "visual_intent": caption,
        "additional_info": {
            "rounded_ratio": aspect_ratio
        },
        "max_critic_rounds": max_critic_rounds  # Add critic rounds control
    }
    
    # Create num_copies identical inputs, each with a unique identifier
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = i
        inputs.append(input_copy)
    
    return inputs

async def process_parallel_candidates(data_list, exp_mode="dev_planner_critic", retrieval_setting="auto", model_name=""):
    """Process multiple candidates in parallel using PaperVizProcessor."""
    # Create experiment config
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        model_name=model_name,
        work_dir=Path(__file__).parent,
    )
    
    # Initialize processor with all agents
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )
    
    # Process all candidates in parallel (concurrency controlled by processor)
    results = []
    concurrent_num = 10  # Process all 10 in parallel
    
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=concurrent_num, do_eval=False
    ):
        results.append(result_data)
    
    return results

async def refine_image_with_nanoviz(image_bytes, edit_prompt, aspect_ratio="21:9", image_size="2K"):
    """
    Refine an image using an Image Editing API.
    
    Args:
        image_bytes: Image data in bytes
        edit_prompt: Text description of desired changes
        aspect_ratio: Output aspect ratio (21:9, 16:9, 3:2)
        image_size: Output resolution (2K or 4K)
    
    Returns:
        Tuple of (edited_image_bytes, success_message)
    """
    try:
        from google import genai
        from google.genai import types
        
        # Initialize client
        project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
        location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
        
        client = genai.Client(vertexai=True, project=project_id, location=location)
        
        # Prepare content
        contents = [
            types.Part.from_text(text=edit_prompt),
            types.Part.from_bytes(
                mime_type="image/jpeg",
                data=image_bytes
            )
        ]
        
        # Configure generation
        config = types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=8192,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
        )
        
        # Generate refined image
        image_model = get_config_val("defaults", "image_model_name", "IMAGE_MODEL_NAME", "")
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=image_model,
            contents=contents,
            config=config
        )
        
        # Extract image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    edited_image_data = part.inline_data.data
                    
                    if isinstance(edited_image_data, bytes):
                        return edited_image_data, "✅ Image refined successfully!"
                    elif isinstance(edited_image_data, str):
                        return base64.b64decode(edited_image_data), "✅ Image refined successfully!"
        
        return None, "❌ No image data found in response"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def ensure_minimal_data_tree():
    """
    Ensure minimal data tree exists so retrieval='none' and demo runs don't fail
    when users have not downloaded PaperBananaBench.
    """
    data_root = Path(__file__).parent / "data" / "PaperBananaBench"
    for task in ("diagram", "plot"):
        task_dir = data_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        ref_path = task_dir / "ref.json"
        if not ref_path.exists():
            ref_path.write_text("[]", encoding="utf-8")


def slugify(value, max_len=80):
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        value = "untitled"
    return value[:max_len].strip("-")


def get_final_candidate_keys(result, exp_mode, task_name="diagram"):
    """
    Return (final_image_key, final_desc_key) for one candidate result.
    """
    final_image_key = None
    final_desc_key = None

    for round_idx in range(3, -1, -1):
        image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if image_key in result and result[image_key]:
            final_image_key = image_key
            final_desc_key = f"target_{task_name}_critic_desc{round_idx}"
            break

    if not final_image_key:
        if exp_mode == "demo_full":
            final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_stylist_desc0"
        else:
            final_image_key = f"target_{task_name}_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_desc0"

    return final_image_key, final_desc_key


async def discover_figure_briefs_async(markdown_text, paper_title, max_figures=6, model_name=""):
    model_name = model_name or get_config_val("defaults", "model_name", "MODEL_NAME", "")
    return await discover_figure_briefs(
        markdown_text=markdown_text,
        paper_title=paper_title,
        model_name=model_name,
        max_figures=max_figures,
        max_sections=8,
    )


async def generate_candidates_for_briefs_async(
    approved_briefs,
    exp_mode,
    num_candidates,
    aspect_ratio,
    max_critic_rounds,
    model_name="",
):
    all_outputs = []
    for brief in approved_briefs:
        data_list = create_sample_inputs(
            method_content=brief["source_excerpt"],
            caption=brief["caption_final"],
            aspect_ratio=aspect_ratio,
            num_copies=num_candidates,
            max_critic_rounds=max_critic_rounds,
        )
        for idx, item in enumerate(data_list):
            item["filename"] = f"{slugify(brief['title'])}_candidate_{idx}"
            item["figure_title"] = brief["title"]
            item["source_section_title"] = brief.get("source_section_title", "")

        results = await process_parallel_candidates(
            data_list=data_list,
            exp_mode=exp_mode,
            retrieval_setting="none",
            model_name=model_name,
        )
        all_outputs.append({"brief": brief, "results": results})
    return all_outputs


def persist_paper_run_outputs(
    paper_title,
    source_file_name,
    approved_briefs,
    batch_outputs,
    generation_settings,
):
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_slug = slugify(paper_title or "paper")
    base_dir = Path(__file__).parent / "results" / "user_papers" / paper_slug / run_timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "paper_title": paper_title,
        "source_file_name": source_file_name,
        "run_timestamp": run_timestamp,
        "settings": generation_settings,
        "briefs": [],
    }

    for output_idx, item in enumerate(batch_outputs):
        brief = item["brief"]
        results = item["results"]

        figure_slug = slugify(brief.get("title", f"figure-{output_idx + 1}"))
        figure_dir = base_dir / f"{output_idx + 1:02d}_{figure_slug}"
        figure_dir.mkdir(parents=True, exist_ok=True)

        # Persist raw structured outputs.
        result_json_path = figure_dir / "results.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        candidate_png_paths = []
        for candidate_id, result in enumerate(results):
            final_image_key, _ = get_final_candidate_keys(
                result=result,
                exp_mode=generation_settings.get("exp_mode", "demo_planner_critic"),
                task_name="diagram",
            )
            if not final_image_key:
                continue
            img = base64_to_image(result.get(final_image_key))
            if img is None:
                continue
            candidate_png = figure_dir / f"candidate_{candidate_id}.png"
            img.save(candidate_png, format="PNG")
            candidate_png_paths.append(str(candidate_png.relative_to(base_dir)))

        run_metadata["briefs"].append(
            {
                "brief_id": brief.get("brief_id", f"brief_{output_idx + 1}"),
                "title": brief.get("title", ""),
                "source_section_title": brief.get("source_section_title", ""),
                "caption_final": brief.get("caption_final", ""),
                "result_json": str(result_json_path.relative_to(base_dir)),
                "candidate_png_paths": candidate_png_paths,
            }
        )

    metadata_path = base_dir / "generation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    return base_dir, metadata_path


def get_evolution_stages(result, exp_mode):
    """Extract all evolution stages (images and descriptions) from the result."""
    task_name = "diagram"
    stages = []
    
    # Stage 1: Planner output
    planner_img_key = f"target_{task_name}_desc0_base64_jpg"
    planner_desc_key = f"target_{task_name}_desc0"
    if planner_img_key in result and result[planner_img_key]:
        stages.append({
            "name": "📋 Planner",
            "image_key": planner_img_key,
            "desc_key": planner_desc_key,
            "description": "Initial diagram plan based on method content"
        })
    
    # Stage 2: Stylist output (only for demo_full)
    if exp_mode == "demo_full":
        stylist_img_key = f"target_{task_name}_stylist_desc0_base64_jpg"
        stylist_desc_key = f"target_{task_name}_stylist_desc0"
        if stylist_img_key in result and result[stylist_img_key]:
            stages.append({
                "name": "✨ Stylist",
                "image_key": stylist_img_key,
                "desc_key": stylist_desc_key,
                "description": "Stylistically refined description"
            })
    
    # Stage 3+: Critic iterations
    for round_idx in range(4):  # Check up to 4 rounds
        critic_img_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        critic_desc_key = f"target_{task_name}_critic_desc{round_idx}"
        critic_sugg_key = f"target_{task_name}_critic_suggestions{round_idx}"
        
        if critic_img_key in result and result[critic_img_key]:
            stages.append({
                "name": f"🔍 Critic Round {round_idx}",
                "image_key": critic_img_key,
                "desc_key": critic_desc_key,
                "suggestions_key": critic_sugg_key,
                "description": f"Refined after critic feedback (iteration {round_idx})"
            })
    
    return stages

def display_candidate_result(result, candidate_id, exp_mode, key_prefix="candidate"):
    """Display a single candidate result."""
    task_name = "diagram"
    
    final_image_key, final_desc_key = get_final_candidate_keys(
        result=result,
        exp_mode=exp_mode,
        task_name=task_name,
    )
    
    # Display the final image
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.image(img, use_container_width=True, caption=f"Candidate {candidate_id} (Final)")
            
            # Add download button
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="⬇️ Download",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_{key_prefix}_{candidate_id}",
                use_container_width=True
            )
        else:
            st.error(f"Failed to decode image for Candidate {candidate_id}")
    else:
        st.warning(f"No image generated for Candidate {candidate_id}")
    
    # Show evolution timeline in an expander
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
        with st.expander(f"🔄 View Evolution Timeline ({len(stages)} stages)", expanded=False):
            st.caption("See how the diagram evolved through different pipeline stages")
            
            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])
                
                # Display the image for this stage
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, use_container_width=True)
                
                # Show description
                if stage['desc_key'] in result:
                    with st.expander(f"📝 Description", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)
                
                # Show critic suggestions if available
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 Critic Suggestions", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ No changes needed - iteration stopped.")
                        else:
                            st.write(cleaned_sugg)
                
                # Add separator between stages (except for the last one)
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # If only one stage, show description in simpler expander
        with st.expander(f"📝 View Description", expanded=False):
            if final_desc_key and final_desc_key in result:
                # Clean the text to remove invalid UTF-8 characters
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("No description available")

def main():
    ensure_minimal_data_tree()
    st.title("🍌 PaperVizAgent Demo")
    st.markdown("AI-powered scientific diagram generation and refinement")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Generate Candidates", "✨ Refine Image", "📄 Paper Upload"])
    
    # ==================== TAB 1: Generate Candidates ====================
    with tab1:
        st.markdown("### Generate multiple diagram candidates from your method section and caption")
        
        # Sidebar configuration for Tab 1
        with st.sidebar:
            st.title("⚙️ Generation Settings")
            
            exp_mode = st.selectbox(
                "Pipeline Mode",
                ["demo_planner_critic", "demo_full"],
                index=0,
                key="tab1_exp_mode",
                help="Select which agent pipeline to use"
            )
            
            mode_info = {
                "demo_planner_critic": "Planner → Visualizer → Critic → Visualizer",
                "demo_full": "Retriever → Planner → Stylist → Visualizer → Critic → Visualizer. (The stylist can make the diagram more aesthetically pleasing, but prone to be overly simplied. So we recommend trying both modes and select the best one)"
            }
            st.info(f"**Pipeline:** {mode_info[exp_mode]}")
            
            retrieval_setting = st.selectbox(
                "Retrieval Setting",
                ["auto", "manual", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help="How to retrieve reference diagrams: auto (automatic selection), manual (use specified references), random (random selection), none (no retrieval)"
            )
            
            num_candidates = st.number_input(
                "Number of Candidates",
                min_value=1,
                max_value=20,
                value=10,
                key="tab1_num_candidates",
                help="How many parallel candidates to generate"
            )
            
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ["21:9", "16:9", "3:2"],
                key="tab1_aspect_ratio",
                help="Aspect ratio for the generated diagrams"
            )
            
            max_critic_rounds = st.number_input(
                "Max Critic Rounds",
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="Maximum number of critic refinement iterations"
            )
            
            default_model = get_config_val("defaults", "model_name", "MODEL_NAME", "YOUR_MODEL_NAME_HERE")
            options = ["", default_model] if default_model else ["", "YOUR_MODEL_NAME_HERE"]
            
            model_name = st.selectbox(
                "Model Name",
                options,
                index=0,
                key="tab1_model_name",
                help="Model name to use for reasoning"
            )
        
        st.divider()
        
        # Input section
        st.markdown("## 📝 Input")
        
        # Example content
        example_method = r"""## Methodology: The PaperVizAgent Framework
        
        In this section, we present the architecture of PaperVizAgent, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperVizAgent orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive “academic style,” as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts."""

        example_caption = "Figure 1: Overview of our PaperVizAgent framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."
        
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            # Example selector for method content
            method_example = st.selectbox(
                "Load Example (Method)",
                ["None", "PaperVizAgent Framework"],
                key="method_example_selector"
            )
            
            # Set value based on example selection or session state
            if method_example == "PaperVizAgent Framework":
                method_value = example_method
            else:
                method_value = st.session_state.get("method_content", "")
            
            method_content = st.text_area(
                "Method Section Content (Markdown recommended)",
                value=method_value,
                height=250,
                placeholder="Paste the method section content here...",
                help="The method section from the paper that describes the approach. Markdown format is recommended."
            )
        
        with col_input2:
            # Example selector for caption
            caption_example = st.selectbox(
                "Load Example (Caption)",
                ["None", "PaperVizAgent Framework"],
                key="caption_example_selector"
            )
            
            # Set value based on example selection or session state
            if caption_example == "PaperVizAgent Framework":
                caption_value = example_caption
            else:
                caption_value = st.session_state.get("caption", "")
            
            caption = st.text_area(
                "Figure Caption (Markdown recommended)",
                value=caption_value,
                height=250,
                placeholder="Enter the figure caption...",
                help="The caption or description of the figure to generate. Markdown format is recommended."
            )
        
        # Process button
        if st.button("🚀 Generate Candidates", type="primary", use_container_width=True):
            if not method_content or not caption:
                st.error("Please provide both method content and caption!")
            else:
                # Save to session state
                st.session_state["method_content"] = method_content
                st.session_state["caption"] = caption
                
                with st.spinner(f"Generating {num_candidates} candidates in parallel... This may take a few minutes."):
                    # Create input data list
                    input_data_list = create_sample_inputs(
                        method_content=method_content,
                        caption=caption,
                        aspect_ratio=aspect_ratio,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds
                    )
                    
                    # Process in parallel
                    try:
                        results = asyncio.run(process_parallel_candidates(
                            input_data_list, 
                            exp_mode=exp_mode, 
                            retrieval_setting=retrieval_setting,
                            model_name=model_name
                        ))
                        st.session_state["results"] = results
                        st.session_state["exp_mode"] = exp_mode
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str
                        
                        # Save results to JSON file
                        try:
                            # Create results directory if it doesn't exist
                            results_dir = Path(__file__).parent / "results" / "demo"
                            results_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Generate filename with timestamp
                            json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            
                            # Save to JSON with proper encoding handling (like main.py)
                            with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
                                json_string = json.dumps(results, ensure_ascii=False, indent=4)
                                # Clean invalid UTF-8 characters
                                json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
                                f.write(json_string)
                            
                            st.session_state["json_file"] = str(json_filename)
                            st.success(f"✅ Successfully generated {len(results)} candidates!")
                            st.info(f"💾 Results saved to: `{json_filename.name}`")
                        except Exception as e:
                            st.warning(f"⚠️ Generated {len(results)} candidates, but failed to save JSON: {e}")
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display results
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            
            st.divider()
            st.markdown("## 🎨 Generated Candidates")
            st.caption(f"Generated at: {timestamp} | Pipeline: {mode_info.get(current_mode, current_mode)}")
            
            # Show JSON file download if available
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 Results saved to: `{json_file_path.relative_to(Path.cwd())}`")
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            use_container_width=True
                        )
            
            # Display results in a grid (3 columns)
            num_cols = 3
            num_results = len(results)
            
            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)
            
            # Add ZIP download button
            st.divider()
            st.markdown("### 💾 Batch Download")
            
            try:
                import zipfile
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    task_name = "diagram"
                    
                    for candidate_id, result in enumerate(results):
                        final_image_key, _ = get_final_candidate_keys(
                            result=result,
                            exp_mode=current_mode,
                            task_name=task_name,
                        )
                        
                        if final_image_key and final_image_key in result:
                            img = base64_to_image(result[final_image_key])
                            if img:
                                img_buffer = BytesIO()
                                img.save(img_buffer, format="PNG")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.png",
                                    img_buffer.getvalue()
                                )
                
                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"papervizagent_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                st.success("ZIP file ready for download!")
            except Exception as e:
                st.error(f"Failed to create ZIP: {e}")
    
    # ==================== TAB 2: Refine Image ====================
    with tab2:
        st.markdown("### Refine and upscale your diagram to high resolution (2K/4K)")
        st.caption("Upload an image from the candidates or any diagram, describe changes, and generate a high-res version")
        
        # Sidebar for refinement settings
        with st.sidebar:
            st.title("✨ Refinement Settings")
            
            refine_resolution = st.selectbox(
                "Target Resolution",
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help="Higher resolution takes longer but produces better quality"
            )
            
            refine_aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ["21:9", "16:9", "3:2"],
                index=0,
                key="refine_aspect_ratio",
                help="Aspect ratio for the refined image"
            )
        
        st.divider()
        
        # Upload section
        st.markdown("## 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload the diagram you want to refine"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(uploaded_image, use_container_width=True)
            
            with col2:
                st.markdown("### Edit Instructions")
                edit_prompt = st.text_area(
                    "Describe the changes you want",
                    height=200,
                    placeholder="E.g., 'Change the color scheme to match academic paper style' or 'Make the text larger and bolder' or 'Keep everything the same but output in higher resolution'",
                    help="Describe what you want to change or use 'Keep everything the same' for just upscaling",
                    key="edit_prompt"
                )
                
                if st.button("✨ Refine Image", type="primary", use_container_width=True):
                    if not edit_prompt:
                        st.error("Please provide edit instructions!")
                    else:
                        with st.spinner(f"Refining image to {refine_resolution} resolution... This may take a minute."):
                            try:
                                # Convert PIL image to bytes
                                img_byte_arr = BytesIO()
                                uploaded_image.save(img_byte_arr, format='JPEG')
                                image_bytes = img_byte_arr.getvalue()
                                
                                # Call nanoviz API
                                refined_bytes, message = asyncio.run(
                                    refine_image_with_nanoviz(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution
                                    )
                                )
                                
                                if refined_bytes:
                                    st.session_state["refined_image"] = refined_bytes
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"Error during refinement: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            
            # Display refined result if available
            if "refined_image" in st.session_state:
                st.divider()
                st.markdown("## 🎨 Refined Result")
                st.caption(f"Generated at: {st.session_state.get('refine_timestamp', 'N/A')} | Resolution: {refine_resolution}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Before")
                    st.image(uploaded_image, use_container_width=True)
                
                with col2:
                    st.markdown(f"### After ({refine_resolution})")
                    refined_image = Image.open(BytesIO(st.session_state["refined_image"]))
                    st.image(refined_image, use_container_width=True)
                    
                    # Download button
                    st.download_button(
                        label=f"⬇️ Download {refine_resolution} Image",
                        data=st.session_state["refined_image"],
                        file_name=f"refined_{refine_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

    # ==================== TAB 3: Paper Upload ====================
    with tab3:
        st.markdown("### Upload a PDF/DOCX, discover figure briefs, review, and generate diagrams")
        st.caption(
            "Docling-first conversion with fallback extraction. "
            "Figure generation uses retrieval='none' by default."
        )

        default_model = get_config_val("defaults", "model_name", "MODEL_NAME", "")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            paper_title = st.text_input(
                "Paper Title",
                value=st.session_state.get("paper_title", ""),
                key="paper_title_input",
                placeholder="My New Paper",
            )
        with col_cfg2:
            max_briefs = st.number_input(
                "Max Briefs to Discover",
                min_value=1,
                max_value=12,
                value=6,
                key="paper_max_briefs",
            )
        with col_cfg3:
            paper_model_name = st.selectbox(
                "Reasoning Model",
                ["", default_model] if default_model else [""],
                index=1 if default_model else 0,
                key="paper_model_name",
                help="Uses defaults.model_name from config when left empty.",
            )

        st.divider()

        uploaded_doc = st.file_uploader(
            "Upload Paper Document",
            type=["pdf", "docx"],
            key="paper_doc_upload",
            help="Text-based PDFs and DOCX are supported. OCR is not enabled for scanned PDFs.",
        )

        if uploaded_doc is not None:
            st.session_state["paper_title"] = paper_title or Path(uploaded_doc.name).stem
            if st.button("📄 Convert Document to Markdown", type="primary", key="paper_convert_btn"):
                try:
                    upload_dir = Path(__file__).parent / "results" / "user_papers" / "_uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_name = f"{ts}_{slugify(Path(uploaded_doc.name).stem)}{Path(uploaded_doc.name).suffix.lower()}"
                    saved_path = upload_dir / saved_name
                    with open(saved_path, "wb") as f:
                        f.write(uploaded_doc.getvalue())

                    conversion = convert_document_to_markdown(saved_path)
                    st.session_state["paper_source_path"] = str(saved_path)
                    st.session_state["paper_source_name"] = uploaded_doc.name
                    st.session_state["paper_markdown"] = conversion["markdown"]
                    st.session_state["paper_markdown_engine"] = conversion["engine"]
                    st.session_state["paper_markdown_warnings"] = conversion.get("warnings", [])
                    # Reset downstream states for new document
                    st.session_state.pop("paper_discovered_briefs", None)
                    st.session_state.pop("paper_batch_outputs", None)
                    st.success(f"Converted document using `{conversion['engine']}`.")
                except Exception as e:
                    st.error(f"Document conversion failed: {e}")

        if st.session_state.get("paper_markdown"):
            engine_name = st.session_state.get("paper_markdown_engine", "unknown")
            st.info(f"Extraction engine: `{engine_name}`")
            for warn in st.session_state.get("paper_markdown_warnings", []):
                st.warning(warn)

            edited_markdown = st.text_area(
                "Extracted Markdown (editable)",
                value=st.session_state.get("paper_markdown", ""),
                height=320,
                key="paper_markdown_editor",
            )
            st.session_state["paper_markdown"] = edited_markdown

            # Quick visibility into sections used by discovery.
            sections = extract_sections(edited_markdown)
            ranked_sections = rank_sections_for_figure_discovery(sections, max_sections=8)
            with st.expander("🔎 Preview Ranked Sections for Discovery", expanded=False):
                if not ranked_sections:
                    st.caption("No sections detected yet.")
                else:
                    for idx, section in enumerate(ranked_sections, start=1):
                        st.markdown(
                            f"**{idx}. {section['title']}** "
                            f"(score={section['score']:.1f}, words={section['word_count']})"
                        )

            if st.button("🧠 Discover Figure Briefs", key="paper_discover_btn", use_container_width=True):
                resolved_model_name = paper_model_name or default_model
                if not resolved_model_name:
                    st.error(
                        "No reasoning model configured. Set `defaults.model_name` in "
                        "`configs/model_config.yaml` or select a model in the UI."
                    )
                else:
                    with st.spinner("Discovering candidate figure briefs..."):
                        try:
                            discovery_res = asyncio.run(
                                discover_figure_briefs_async(
                                    markdown_text=edited_markdown,
                                    paper_title=st.session_state.get("paper_title", "Untitled Paper"),
                                    max_figures=max_briefs,
                                    model_name=resolved_model_name,
                                )
                            )
                            st.session_state["paper_discovery_raw_response"] = discovery_res.get("raw_response", "")
                            st.session_state["paper_discovered_briefs"] = discovery_res.get("briefs", [])
                            st.success(
                                f"Discovered {len(st.session_state['paper_discovered_briefs'])} candidate figure briefs."
                            )
                        except Exception as e:
                            st.error(f"Figure discovery failed: {e}")

        discovered_briefs = st.session_state.get("paper_discovered_briefs", [])
        if discovered_briefs:
            st.divider()
            st.markdown("## ✅ Mandatory Review Before Generation")
            st.caption("Edit each brief and explicitly approve the ones you want to generate.")

            reviewed_briefs = []
            for idx, brief in enumerate(discovered_briefs):
                brief_key = brief.get("brief_id", f"brief_{idx + 1}")
                with st.expander(f"{idx + 1}. {brief.get('title', 'Untitled Brief')}", expanded=False):
                    brief_title = st.text_input(
                        "Figure Title",
                        value=brief.get("title", ""),
                        key=f"{brief_key}_title",
                    )
                    source_section_title = st.text_input(
                        "Source Section",
                        value=brief.get("source_section_title", ""),
                        key=f"{brief_key}_section",
                    )
                    source_excerpt = st.text_area(
                        "Source Excerpt",
                        value=brief.get("source_excerpt", ""),
                        height=180,
                        key=f"{brief_key}_excerpt",
                    )
                    caption_final = st.text_area(
                        "Caption (editable final)",
                        value=brief.get("caption_draft", ""),
                        height=120,
                        key=f"{brief_key}_caption",
                    )
                    approval = st.checkbox(
                        "Approve this brief for generation",
                        value=False,
                        key=f"{brief_key}_approved",
                    )

                    reviewed_briefs.append(
                        {
                            "brief_id": brief_key,
                            "title": brief_title.strip(),
                            "source_section_title": source_section_title.strip(),
                            "source_excerpt": source_excerpt.strip(),
                            "caption_final": caption_final.strip(),
                            "approved": approval,
                        }
                    )

            st.session_state["paper_reviewed_briefs"] = reviewed_briefs
            approved_briefs = [b for b in reviewed_briefs if b["approved"]]
            st.info(f"Approved briefs: {len(approved_briefs)} / {len(reviewed_briefs)}")

            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            with col_g1:
                paper_exp_mode = st.selectbox(
                    "Pipeline Mode",
                    ["demo_planner_critic", "demo_full"],
                    index=0,
                    key="paper_exp_mode",
                )
            with col_g2:
                paper_num_candidates = st.number_input(
                    "Candidates per Brief",
                    min_value=1,
                    max_value=10,
                    value=4,
                    key="paper_num_candidates",
                )
            with col_g3:
                paper_aspect_ratio = st.selectbox(
                    "Aspect Ratio",
                    ["21:9", "16:9", "3:2"],
                    index=1,
                    key="paper_aspect_ratio",
                )
            with col_g4:
                paper_max_critic_rounds = st.number_input(
                    "Max Critic Rounds",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key="paper_max_critic_rounds",
                )

            if st.button("🚀 Generate Approved Briefs", type="primary", key="paper_generate_btn", use_container_width=True):
                if not approved_briefs:
                    st.error("Approve at least one brief before generation.")
                elif not (paper_model_name or default_model):
                    st.error(
                        "No reasoning model configured. Set `defaults.model_name` in "
                        "`configs/model_config.yaml` or select a model in the UI."
                    )
                else:
                    with st.spinner(
                        f"Generating {paper_num_candidates} candidates each for {len(approved_briefs)} approved briefs..."
                    ):
                        try:
                            batch_outputs = asyncio.run(
                                generate_candidates_for_briefs_async(
                                    approved_briefs=approved_briefs,
                                    exp_mode=paper_exp_mode,
                                    num_candidates=paper_num_candidates,
                                    aspect_ratio=paper_aspect_ratio,
                                    max_critic_rounds=paper_max_critic_rounds,
                                    model_name=paper_model_name or default_model,
                                )
                            )
                            st.session_state["paper_batch_outputs"] = batch_outputs

                            run_dir, metadata_path = persist_paper_run_outputs(
                                paper_title=st.session_state.get("paper_title", "Untitled Paper"),
                                source_file_name=st.session_state.get("paper_source_name", ""),
                                approved_briefs=approved_briefs,
                                batch_outputs=batch_outputs,
                                generation_settings={
                                    "exp_mode": paper_exp_mode,
                                    "num_candidates": paper_num_candidates,
                                    "aspect_ratio": paper_aspect_ratio,
                                    "max_critic_rounds": paper_max_critic_rounds,
                                    "retrieval_setting": "none",
                                    "model_name": paper_model_name or default_model,
                                },
                            )
                            st.session_state["paper_run_dir"] = str(run_dir)
                            st.session_state["paper_metadata_path"] = str(metadata_path)
                            st.success("Generation completed for approved briefs.")
                        except Exception as e:
                            st.error(f"Batch generation failed: {e}")

        batch_outputs = st.session_state.get("paper_batch_outputs", [])
        if batch_outputs:
            st.divider()
            st.markdown("## 🎨 Generated Figures from Uploaded Paper")
            if st.session_state.get("paper_run_dir"):
                st.info(f"Saved run folder: `{st.session_state['paper_run_dir']}`")

            metadata_path = st.session_state.get("paper_metadata_path")
            if metadata_path and Path(metadata_path).exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download Generation Metadata (JSON)",
                        data=f.read(),
                        file_name=Path(metadata_path).name,
                        mime="application/json",
                        key="paper_download_metadata",
                    )

            current_mode = st.session_state.get("paper_exp_mode", "demo_planner_critic")
            for brief_idx, item in enumerate(batch_outputs):
                brief = item["brief"]
                results = item["results"]
                st.markdown(f"### {brief_idx + 1}. {brief.get('title', 'Untitled Figure')}")
                st.caption(brief.get("caption_final", ""))

                num_cols = 3
                for row_start in range(0, len(results), num_cols):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        result_idx = row_start + col_idx
                        if result_idx >= len(results):
                            continue
                        with cols[col_idx]:
                            display_candidate_result(
                                results[result_idx],
                                candidate_id=result_idx,
                                exp_mode=current_mode,
                                key_prefix=f"paper_{brief_idx}",
                            )

if __name__ == "__main__":
    main()
