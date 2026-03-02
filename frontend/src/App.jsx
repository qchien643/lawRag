import { useState, useEffect, useCallback } from 'react';
import { listFiles, clearAll as apiClearAll, checkHealth } from './api';
import FileUpload from './components/FileUpload';
import FileList from './components/FileList';
import ChatPanel from './components/ChatPanel';
import './App.css';

export default function App() {
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState(new Set());
  const [online, setOnline] = useState(false);

  const refreshFiles = useCallback(async () => {
    try {
      const data = await listFiles();
      setFiles(data.files || []);
    } catch {
      setFiles([]);
    }
  }, []);

  useEffect(() => {
    refreshFiles();
    const checkStatus = async () => {
      try {
        const d = await checkHealth();
        setOnline(d.status === 'ok');
      } catch {
        setOnline(false);
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, [refreshFiles]);

  function toggleFile(name) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  function selectAll() {
    setSelected(new Set(files.map((f) => f.name)));
  }

  function deselectAll() {
    setSelected(new Set());
  }

  async function handleClear() {
    if (!confirm('Xoá toàn bộ dữ liệu? Không thể hoàn tác.')) return;
    try {
      await apiClearAll();
      setSelected(new Set());
      refreshFiles();
    } catch { }
  }

  return (
    <>
      <header className="header">
        <div className="logo">R</div>
        <h1 className="title">RAG Legal Assistant</h1>
        <div className={`status-dot ${online ? '' : 'offline'}`}
          title={online ? 'Online' : 'Offline'} />
      </header>

      <main className="main-layout">
        <aside className="sidebar">
          <div className="section-title">📤 Upload tài liệu</div>
          <FileUpload onUploaded={refreshFiles} />

          <div className="section-title">📂 Tài liệu trong database</div>
          <FileList
            files={files}
            selected={selected}
            onToggle={toggleFile}
            onSelectAll={selectAll}
            onDeselectAll={deselectAll}
            onClear={handleClear}
            onRefresh={refreshFiles}
          />
        </aside>

        <ChatPanel selectedFiles={selected} />
      </main>
    </>
  );
}
