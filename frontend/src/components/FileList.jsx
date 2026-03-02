import { deleteFile as apiDeleteFile } from '../api';
import './FileList.css';

export default function FileList({ files, selected, onToggle, onSelectAll, onDeselectAll, onClear, onRefresh }) {

    async function handleDelete(name) {
        if (!confirm(`Xoá file "${name}" khỏi database?`)) return;
        try {
            await apiDeleteFile(name);
            onRefresh?.();
        } catch { }
    }

    return (
        <div className="filelist-section">
            {files.length === 0 ? (
                <div className="empty-state">
                    <div className="empty-icon">📭</div>
                    <p>Chưa có tài liệu nào.<br />Hãy upload file để bắt đầu.</p>
                </div>
            ) : (
                <div className="file-items">
                    {files.map((f) => (
                        <div
                            key={f.name}
                            className={`file-item ${selected.has(f.name) ? 'selected' : ''}`}
                            onClick={() => onToggle(f.name)}
                        >
                            <div className="checkbox" />
                            <div className="file-icon">📄</div>
                            <div className="file-info">
                                <div className="file-name" title={f.name}>{f.name}</div>
                                <div className="file-meta">{f.chunk_count} chunks</div>
                            </div>
                            <button
                                className="delete-btn"
                                onClick={(e) => { e.stopPropagation(); handleDelete(f.name); }}
                                title="Xoá"
                            >
                                🗑
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {files.length > 0 && (
                <div className="sidebar-actions">
                    <button onClick={onSelectAll}>Chọn tất cả</button>
                    <button onClick={onDeselectAll}>Bỏ chọn</button>
                    <button className="danger" onClick={onClear}>Xoá tất cả</button>
                </div>
            )}
        </div>
    );
}
