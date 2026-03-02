const API_BASE = '/api';

export async function checkHealth() {
    const r = await fetch(`${API_BASE}/health`);
    return r.json();
}

export async function uploadFiles(files) {
    const formData = new FormData();
    for (const f of files) formData.append('files', f);
    const r = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
    if (!r.ok) {
        const err = await r.json();
        throw new Error(err.detail || 'Upload failed');
    }
    return r.json();
}

export async function listFiles() {
    const r = await fetch(`${API_BASE}/files`);
    return r.json();
}

export async function deleteFile(name) {
    const r = await fetch(`${API_BASE}/files/${encodeURIComponent(name)}`, { method: 'DELETE' });
    return r.json();
}

export async function queryRAG(question, sourceFilter) {
    const body = { question };
    if (sourceFilter && sourceFilter.length > 0) {
        body.source_filter = sourceFilter;
    }
    const r = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!r.ok) {
        const err = await r.json();
        throw new Error(err.detail || 'Query failed');
    }
    return r.json();
}

export async function clearAll() {
    const r = await fetch(`${API_BASE}/clear`, { method: 'DELETE' });
    return r.json();
}
