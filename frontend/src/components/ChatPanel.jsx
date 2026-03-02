import { useState, useRef, useEffect } from 'react';
import { queryRAG } from '../api';
import './ChatPanel.css';

export default function ChatPanel({ selectedFiles }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEnd = useRef(null);
    const textareaRef = useRef(null);

    useEffect(() => {
        messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    async function handleSend() {
        const q = input.trim();
        if (!q || loading) return;

        setMessages((prev) => [...prev, { role: 'user', text: q }]);
        setInput('');
        if (textareaRef.current) textareaRef.current.style.height = 'auto';
        setLoading(true);

        try {
            const filter = selectedFiles.size > 0 ? Array.from(selectedFiles) : null;
            const data = await queryRAG(q, filter);
            setMessages((prev) => [...prev, { role: 'assistant', text: data.answer }]);
        } catch (err) {
            setMessages((prev) => [...prev, { role: 'assistant', text: `⚠️ ${err.message}` }]);
        }
        setLoading(false);
    }

    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    }

    function handleInput(e) {
        setInput(e.target.value);
        e.target.style.height = 'auto';
        e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
    }

    const filterText =
        selectedFiles.size > 0
            ? `🔍 Tìm kiếm trong ${selectedFiles.size} file được chọn`
            : '💡 Chọn file ở sidebar để lọc, hoặc tìm trong tất cả';

    return (
        <div className="chat-area">
            <div className="messages">
                {messages.length === 0 && !loading && (
                    <div className="welcome">
                        <div className="welcome-icon">⚖️</div>
                        <h2>Hỏi đáp tài liệu pháp luật</h2>
                        <p>Upload tài liệu PDF, chọn file muốn tra cứu, rồi đặt câu hỏi.</p>
                    </div>
                )}

                {messages.map((m, i) => (
                    <div key={i} className={`message ${m.role}`}>
                        {m.text}
                    </div>
                ))}

                {loading && (
                    <div className="message thinking">
                        <span /><span /><span />
                    </div>
                )}
                <div ref={messagesEnd} />
            </div>

            <div className="input-area">
                <div className="input-wrapper">
                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={handleInput}
                        onKeyDown={handleKeyDown}
                        placeholder="Nhập câu hỏi tại đây..."
                        rows={1}
                        disabled={loading}
                    />
                    <button onClick={handleSend} disabled={loading || !input.trim()} title="Gửi">
                        ➤
                    </button>
                </div>
                <div className="selected-hint">{filterText}</div>
            </div>
        </div>
    );
}
