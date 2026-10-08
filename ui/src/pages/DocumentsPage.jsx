import { useState } from "react";
import { api } from "../services/api";
import {
  UploadCloud,
  FileText,
  Search,
  CheckCircle,
  AlertCircle,
  Clock,
  Layers,
} from "lucide-react";

export default function DocumentsPage() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState("");

  const [query, setQuery] = useState("");
  const [querying, setQuerying] = useState(false);
  const [queryResult, setQueryResult] = useState(null);
  const [queryError, setQueryError] = useState("");

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setUploadResult(null);
    setUploadError("");

    try {
      const res = await api.uploadFile(file);
      setUploadResult(res);
      setFile(null);
    } catch (err) {
      setUploadError(err.message || "File upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setQuerying(true);
    setQueryResult(null);
    setQueryError("");

    try {
      const res = await api.queryDocuments(query.trim());
      setQueryResult(res);
    } catch (err) {
      setQueryError(err.message || "Query failed.");
    } finally {
      setQuerying(false);
    }
  };

  return (
    <div className="page-documents">
      <div className="page-header">
        <h1>Document Intelligence & RAG</h1>
        <p>Index documents, spreadsheets, slides, and media into the Chroma vector database</p>
      </div>

      <div className="grid-2-col">
        {/* Upload Card */}
        <div className="panel-card">
          <div className="panel-header">
            <UploadCloud className="icon-accent" size={22} />
            <h2>Upload & Index Documents</h2>
          </div>
          <p className="panel-desc">
            Supports PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), Images (PNG/JPG OCR), and Audio/Video.
          </p>

          <form onSubmit={handleUpload} className="upload-form">
            <div className="file-dropzone">
              <input
                type="file"
                id="file-upload"
                onChange={(e) => setFile(e.target.files[0] || null)}
                disabled={uploading}
              />
              <label htmlFor="file-upload">
                <FileText size={32} className="dropzone-icon" />
                <span className="dropzone-text">
                  {file ? file.name : "Click or drag file here to upload"}
                </span>
                <span className="dropzone-sub">Max size: 25 MB</span>
              </label>
            </div>

            {uploadError && (
              <div className="auth-alert error">
                <AlertCircle size={18} />
                <span>{uploadError}</span>
              </div>
            )}

            {uploadResult && (
              <div className="auth-alert success">
                <CheckCircle size={18} />
                <span>
                  Indexed "{uploadResult.filename}" ({uploadResult.chunks_indexed} chunks created).
                </span>
              </div>
            )}

            <button type="submit" className="btn-primary full-width" disabled={!file || uploading}>
              {uploading ? "Ingesting & Embedding..." : "Upload & Index"}
            </button>
          </form>
        </div>

        {/* Direct Query Card */}
        <div className="panel-card">
          <div className="panel-header">
            <Search className="icon-accent" size={22} />
            <h2>Direct Document Search</h2>
          </div>
          <p className="panel-desc">
            Query the vector database directly without activating full conversational state.
          </p>

          <form onSubmit={handleQuery} className="query-form">
            <div className="form-group">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What does the document say about..."
                disabled={querying}
              />
            </div>

            <button type="submit" className="btn-secondary full-width" disabled={!query.trim() || querying}>
              {querying ? "Searching Vector Store..." : "Run Vector Search"}
            </button>
          </form>

          {queryError && (
            <div className="auth-alert error" style={{ marginTop: "1rem" }}>
              <AlertCircle size={18} />
              <span>{queryError}</span>
            </div>
          )}

          {queryResult && (
            <div className="query-result-box">
              <div className="result-meta">
                <span>Retrieved: {queryResult.retrieved_chunks} Chunks</span>
                {queryResult.route && <span>Route: {queryResult.route}</span>}
              </div>
              <div className="result-content">{queryResult.answer}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
