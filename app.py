import os
import shutil
import gradio as gr

import config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_chain import RAGChain

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
processor = DocumentProcessor()
vs_manager = VectorStoreManager()
rag_chain = RAGChain(vs_manager)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Backend functions
# ---------------------------------------------------------------------------
def process_files(files):
    """Upload and process files into the vector store."""
    if not files:
        return "Vui long chon it nhat mot file."

    file_paths = []
    for f in files:
        file_name = os.path.basename(f)
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in config.SUPPORTED_EXTENSIONS:
            continue
        dest = os.path.join(config.UPLOAD_DIR, file_name)
        if os.path.abspath(f) != os.path.abspath(dest):
            shutil.copy2(f, dest)
        file_paths.append(dest)

    if not file_paths:
        return (
            f"Khong co file hop le. Dinh dang ho tro: "
            f"{', '.join(config.SUPPORTED_EXTENSIONS)}"
        )

    # Process
    try:
        chunks = processor.load_and_process(file_paths)
        vs_manager.add_documents(chunks)
        file_names = [os.path.basename(p) for p in file_paths]
        return (
            f"Da xu ly thanh cong {len(file_paths)} file "
            f"({', '.join(file_names)}). "
            f"Tao duoc {len(chunks)} doan van ban. "
            f"Tong so doan trong co so du lieu: {vs_manager.document_count}."
        )
    except Exception as e:
        return f"Loi khi xu ly file: {str(e)}"


def ask_question(question, history):
    """Process user question through RAG chain."""
    if not question or not question.strip():
        return history, ""

    if not vs_manager.is_ready:
        bot_msg = "Vui long upload va xu ly file truoc khi dat cau hoi."
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": bot_msg},
        ]
        return history, ""

    try:
        answer = rag_chain.query(question.strip())
    except Exception as e:
        answer = f"Loi khi xu ly cau hoi: {str(e)}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


def clear_all():
    """Clear vector store and chat history."""
    vs_manager.clear()
    return [], "Da xoa toan bo du lieu va lich su tro chuyen."


# ---------------------------------------------------------------------------
# Gradio UI  (khong su dung icon)
# ---------------------------------------------------------------------------
CSS = """
    .header { text-align: center; margin-bottom: 8px; }
    .status-box textarea { font-size: 14px !important; }
"""


def build_app():
    with gr.Blocks(title="RAG - Truy van tai lieu") as app:

        # ---- Header ----
        gr.Markdown(
            """
            <div class="header">
            <h1>He thong truy van tren tai lieu (RAG)</h1>
            </div>
            """,
        )

        with gr.Row():
            # ---- Left column: File upload ----
            with gr.Column(scale=1):
                gr.Markdown("### Tai lieu")
                file_input = gr.File(
                    label="Chon file",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".xlsx"],
                    type="filepath",
                )
                process_btn = gr.Button("Xu ly file", variant="primary")
                status_box = gr.Textbox(
                    label="Trang thai",
                    interactive=False,
                    lines=3,
                    elem_classes=["status-box"],
                )
                clear_btn = gr.Button("Xoa toan bo du lieu", variant="stop")

            # ---- Right column: Chat ----
            with gr.Column(scale=2):
                gr.Markdown("### Hoi dap")
                chatbot = gr.Chatbot(
                    label="Lich su tro chuyen",
                    height=460,
                )
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Cau hoi",
                        placeholder="Nhap cau hoi tai day...",
                        scale=5,
                        lines=1,
                    )
                    ask_btn = gr.Button("Gui", variant="primary", scale=1)

        # ---- Events ----
        process_btn.click(
            fn=process_files,
            inputs=[file_input],
            outputs=[status_box],
        )
        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input],
        )
        question_input.submit(
            fn=ask_question,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input],
        )
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[chatbot, status_box],
        )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_port=7860,
        css=CSS
    )
