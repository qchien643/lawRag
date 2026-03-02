import { useState, useRef } from 'react';
import { uploadFiles } from '../api';
import './FileUpload.css';

export default function FileUpload({ onUploaded }) {
    const [status, setStatus] = useState(null); // { type, text }
    const [dragging, setDragging] = useState(false);
    const inputRef = useRef(null);

    async function handleFiles(files) {
        if (!files.length) return;
        setStatus({ type: 'processing', text: `Đang xử lý ${files.length} file...` });
        try {
            const data = await uploadFiles(files);
            setStatus({ type: 'success', text: `✓ ${data.message}` });
            onUploaded?.();
        } catch (err) {
            setStatus({ type: 'error', text: `✗ ${err.message}` });
        }
        setTimeout(() => setStatus(null), 6000);
    }

    function onDrop(e) {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
    }

    return (
        <div className="upload-section">
            <div
                className={`upload-zone ${dragging ? 'drag-over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
            >
                <input
                    ref={inputRef}
                    type="file"
                    multiple
                    accept=".pdf"
                    onChange={(e) => { handleFiles(e.target.files); e.target.value = ''; }}
                    hidden
                />
                <div className="upload-icon">📁</div>
                <p>Kéo thả hoặc click để chọn file</p>
                <span className="upload-hint">Hỗ trợ: PDF</span>
            </div>

            {status && (
                <div className={`upload-status ${status.type}`}>
                    {status.type === 'processing' && <div className="spinner" />}
                    <span>{status.text}</span>
                </div>
            )}
        </div>
    );
}
