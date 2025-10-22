#!/usr/bin/env python
"""REFRAG Anything Studio - Interactive UI for multimodal ingestion and querying.

This Gradio application wraps the :class:`raganything.RAGAnything` pipeline with
the REFRAG runtime enabled by default.  It provides:

* Guided configuration for OpenAI compatible endpoints
* Drag-and-drop document ingestion with progress feedback
* Conversational querying with optional REFRAG compression controls

Run with ``python examples/refrag_ui_app.py`` and open the reported URL.
"""

from __future__ import annotations

import asyncio
import os
import platform
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gradio as gr
from dotenv import load_dotenv

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig


load_dotenv(dotenv_path=".env", override=False)


EMBEDDING_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


def _ensure_event_loop_result(awaitable):
    """Run an awaitable to completion using an isolated event loop."""

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(awaitable)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()


def _env_api_config():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    return api_key, base_url


def _force_cpu_device() -> str:
    """
    Return 'cpu' on macOS (Darwin) or when FORCE_CPU=1 in env; otherwise empty string.
    This is passed through to MinerU to disable MPS use that triggers NotImplementedError.
    """

    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"
    return "cpu" if platform.system() == "Darwin" else ""


def _build_llm_functions(
    api_key: str,
    base_url: Optional[str],
    llm_model: str,
    vision_model: str,
    embedding_model: str,
    embedding_dim: int,
) -> Tuple:
    """Create callable wrappers for OpenAI-compatible endpoints."""

    def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Sequence] = None,
        **kwargs,
    ) -> str:
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=list(history_messages or []),
            api_key=api_key,
            base_url=base_url or None,
            **kwargs,
        )

    def vision_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[Sequence] = None,
        image_data: Optional[str] = None,
        messages: Optional[Sequence] = None,
        **kwargs,
    ) -> str:
        if messages is not None:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=list(messages),
                api_key=api_key,
                base_url=base_url or None,
                **kwargs,
            )

        if image_data is not None:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                },
                            },
                        ],
                    }
                ],
                api_key=api_key,
                base_url=base_url or None,
                **kwargs,
            )

        return llm_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url or None,
        ),
    )

    return llm_model_func, vision_model_func, embedding_func


def _initial_config(
    working_dir: str,
    parser: str,
    parse_method: str,
    refrag_enabled: bool,
    compression_k: int,
    entropy_tau: float,
    expand_numeric: bool,
    expand_legal: bool,
) -> RAGAnythingConfig:
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method=parse_method,
    )

    config.refrag = replace(
        config.refrag,
        enabled=refrag_enabled,
        compression_rate_k=compression_k,
        selective_expand=replace(
            config.refrag.selective_expand,
            entropy_tau=entropy_tau,
            expand_numeric_tables=expand_numeric,
            expand_legal_citations=expand_legal,
        ),
    )

    return config


def initialize_pipeline(
    existing_rag: Optional[RAGAnything],
    working_dir: str,
    parser: str,
    parse_method: str,
    llm_model: str,
    vision_model: str,
    embedding_model: str,
    refrag_enabled: bool,
    compression_k: int,
    entropy_tau: float,
    expand_numeric: bool,
    expand_legal: bool,
) -> Tuple[str, Optional[RAGAnything], List[dict]]:
    """Create or refresh the RAGAnything pipeline."""

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    api_key, base_url = _env_api_config()
    if not api_key:
        return (
            "❌ No OPENAI_API_KEY found in environment (.env). Please set it and reload the app.",
            existing_rag,
            [],
        )

    if existing_rag is not None:
        try:
            _ensure_event_loop_result(existing_rag.finalize_storages())
        except Exception:
            pass

    working_dir = working_dir or "./rag_storage_ui"
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    embedding_dim = EMBEDDING_DIMS.get(embedding_model, 1536)

    llm_model_func, vision_model_func, embedding_func = _build_llm_functions(
        api_key=api_key,
        base_url=base_url,
        llm_model=llm_model,
        vision_model=vision_model or llm_model,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )

    config = _initial_config(
        working_dir=working_dir,
        parser=parser,
        parse_method=parse_method,
        refrag_enabled=refrag_enabled,
        compression_k=compression_k,
        entropy_tau=entropy_tau,
        expand_numeric=expand_numeric,
        expand_legal=expand_legal,
    )

    try:
        config.parser = parser
        setattr(config, "mineru_device", _force_cpu_device())
    except Exception:
        pass

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    return (
        "✅ Pipeline initialized. Upload documents to begin building context.",
        rag,
        [],
    )


def _format_doc_table(documents: Iterable[Dict[str, str]]) -> List[dict]:
    return list(documents)


def process_documents(
    rag: Optional[RAGAnything],
    existing_docs: List[Dict[str, str]],
    files: List[str],
    parse_method: str,
) -> Tuple[str, List[Dict[str, str]]]:
    if rag is None:
        return "❌ Initialize the pipeline before uploading files.", existing_docs

    if not files:
        return "⚠️ No files selected for ingestion.", existing_docs

    new_docs: List[Dict[str, str]] = []
    device = _force_cpu_device()
    kwargs: Dict[str, str] = {}
    if device:
        kwargs["device"] = device
    for file_path in files:
        try:
            _ensure_event_loop_result(
                rag.process_document_complete(
                    file_path,
                    parse_method=parse_method or rag.config.parse_method,
                    **kwargs,
                )
            )
            new_docs.append({"File": Path(file_path).name, "Status": "Processed"})
        except Exception as exc:  # pragma: no cover - UI level reporting
            new_docs.append({"File": Path(file_path).name, "Status": f"Error: {exc}"})

    combined = existing_docs + new_docs
    status_lines = [f"• {doc['File']}: {doc['Status']}" for doc in new_docs]
    status = (
        "✅ Ingestion complete:\n" + "\n".join(status_lines)
        if status_lines
        else "⚠️ No new documents were processed."
    )

    return status, combined


def answer_query(
    rag: Optional[RAGAnything],
    history: List[Dict[str, str]],
    question: str,
    mode: str,
) -> Tuple[List[Dict[str, str]], str, List[Dict[str, str]]]:
    raw_question = question or ""
    question = raw_question.strip()

    if rag is None:
        if question:
            updated = history + [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": "Pipeline not initialized. Please configure and initialize first.",
                },
            ]
            return updated, "", updated
        return history, raw_question, history

    if not question:
        return history, "", history

    try:
        answer = _ensure_event_loop_result(rag.aquery(question, mode=mode))
    except Exception as exc:  # pragma: no cover - UI level reporting
        answer = f"Error: {exc}"

    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": str(answer)},
    ]
    return updated_history, "", updated_history


def clear_conversation() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    return [], []


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="REFRAG Anything Studio",
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="violet",
            neutral_hue="slate",
        ),
        css="""
    .gradio-container { max-width: 1080px !important; margin: auto; }
    .hero-card {
        background: radial-gradient(1200px 400px at 10% 0%, rgba(0,217,255,0.06), transparent),
                    linear-gradient(180deg, rgba(0,0,0,0.35), rgba(0,0,0,0.15));
        border: 1px solid rgba(0, 217, 255, 0.25);
        border-radius: 16px; padding: 18px 20px;
        backdrop-filter: blur(6px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .section-title {
        font-weight: 700; letter-spacing: .3px;
        border-left: 4px solid #00d9ff; padding-left: 10px; margin: 6px 0 12px;
    }
    .wrap-table table { white-space: normal !important; }
    button { border-radius: 10px !important; }
    """
    ) as demo:
        gr.HTML(
            """
<div class="hero-card">
  <h1 style="margin:0 0 6px 0;">REFRAG Anything Studio</h1>
  <div style="opacity:.85">
    <b>Multimodal dual-graph RAG</b> with <b>REFRAG compressed decoding</b>.
    <br/>Credentials are read automatically from <code>.env</code>.
  </div>
</div>
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown('<div class="section-title">Storage & Parsing</div>')
                working_dir = gr.Textbox(
                    label="Working Directory",
                    value="./rag_storage_ui",
                    placeholder="Directory for indexes and caches",
                )
                with gr.Row():
                    parser = gr.Dropdown(
                        ["mineru", "docling"],
                        value="mineru",
                        label="Parser",
                    )
                    parse_method = gr.Dropdown(
                        ["auto", "ocr", "txt"],
                        value="auto",
                        label="Parse Method",
                    )

                gr.Markdown('<div class="section-title">Models & REFRAG</div>')
                llm_model = gr.Textbox(
                    label="LLM Model",
                    value="gpt-4o-mini",
                )
                vision_model = gr.Textbox(
                    label="Vision Model",
                    value="gpt-4o",
                )
                embedding_model = gr.Textbox(
                    label="Embedding Model",
                    value="text-embedding-3-large",
                )
                refrag_enabled = gr.Checkbox(
                    label="Enable REFRAG Compression",
                    value=True,
                )
                compression_k = gr.Slider(
                    label="Compression Rate (k)",
                    value=16,
                    minimum=4,
                    maximum=64,
                    step=4,
                )
                entropy_tau = gr.Slider(
                    label="Entropy Threshold (τ)",
                    value=2.0,
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                )
                expand_numeric = gr.Checkbox(
                    label="Auto-expand numeric & table-heavy chunks",
                    value=True,
                )
                expand_legal = gr.Checkbox(
                    label="Auto-expand legal & citation chunks",
                    value=True,
                )

                init_button = gr.Button(
                    "Initialize Pipeline",
                    variant="primary",
                    icon="🚀",
                )
            with gr.Column(scale=1):
                env_ok = "✅ .env loaded" if os.getenv("OPENAI_API_KEY") else "⚠️ .env missing OPENAI_API_KEY"
                gr.Markdown(f"**Env Status:** {env_ok}")
                status = gr.Markdown(
                    "Configure settings and click **Initialize Pipeline**. API credentials are read from your environment (.env).",
                )
                doc_table = gr.Dataframe(
                    headers=["File", "Status"],
                    datatype=["str", "str"],
                    label="Ingested Documents",
                    wrap=True,
                    interactive=False,
                    elem_classes=["wrap-table"],
                )

        pipeline_state = gr.State(None)
        documents_state = gr.State([])
        chat_state = gr.State([])

        init_click = init_button.click(
            initialize_pipeline,
            inputs=[
                pipeline_state,
                working_dir,
                parser,
                parse_method,
                llm_model,
                vision_model,
                embedding_model,
                refrag_enabled,
                compression_k,
                entropy_tau,
                expand_numeric,
                expand_legal,
            ],
            outputs=[status, pipeline_state, documents_state],
        )

        init_click.then(
            lambda docs: gr.update(value=_format_doc_table(docs)),
            inputs=documents_state,
            outputs=doc_table,
        )

        with gr.Tab("Ingest"):
            file_uploader = gr.File(
                label="Upload Documents",
                file_count="multiple",
                type="filepath",
            )
            process_button = gr.Button("Process Selected Files", variant="primary")

            process_button.click(
                process_documents,
                inputs=[pipeline_state, documents_state, file_uploader, parse_method],
                outputs=[status, documents_state],
            ).then(
                lambda docs: gr.update(value=_format_doc_table(docs)),
                inputs=documents_state,
                outputs=doc_table,
            )

        with gr.Tab("Chat"):
            chatbox = gr.Chatbot(label="Conversation", type="messages")
            question = gr.Textbox(
                label="Ask a question",
                placeholder="What would you like to know about your documents?",
            )
            mode = gr.Dropdown(
                ["mix", "local", "global", "hybrid", "naive", "bypass"],
                value="mix",
                label="Retrieval Mode",
            )
            ask_button = gr.Button("Ask", variant="primary")
            clear_button = gr.Button("Clear Conversation")

            ask_button.click(
                answer_query,
                inputs=[pipeline_state, chat_state, question, mode],
                outputs=[chatbox, question, chat_state],
            )

            clear_button.click(
                clear_conversation,
                outputs=[chatbox, chat_state],
            )

        demo.load(
            lambda docs: gr.update(value=_format_doc_table(docs)),
            inputs=documents_state,
            outputs=doc_table,
        )

    return demo


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
