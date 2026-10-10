import { useState } from "react";
import { Copy, Check, Info, AlertTriangle, Lightbulb, CheckCircle2, ShieldAlert } from "lucide-react";
import { resolveImageUrl } from "../services/api";

function sanitizeContent(raw) {
  if (!raw) return "";
  let text = typeof raw === "string" ? raw : String(raw);

  // Strip SystemMessage / AIMessage multiline wrappers
  const msgMatch = text.match(
    /^(?:SystemMessage|AIMessage|HumanMessage|ChatMessage)\s*\n+content=(?:"|'|"""|''')(.*?)(?:"|'|"""|''')(?:[\s\S]*)$/s
  );
  if (msgMatch) {
    text = msgMatch[1];
  }

  // Strip SystemMessage(...) / AIMessage(...)
  const parenMatch = text.match(
    /^(?:SystemMessage|AIMessage|HumanMessage|ChatMessage)\s*\(\s*(?:content=)?(['"]{1,3})(.*?)\1(?:\s*,\s*response_metadata=.*)?\s*\)$/s
  );
  if (parenMatch) {
    text = parenMatch[2];
  }

  // Strip trailing response_metadata / additional_kwargs blocks
  text = text.replace(/\n+response_metadata\s*\{[\s\S]*$/m, "");
  text = text.replace(/\n+additional_kwargs\s*\{[\s\S]*$/m, "");

  return text.trim();
}

export default function MarkdownRenderer({ content, onImageClick }) {
  if (!content) return null;

  const cleanContent = sanitizeContent(content);

  // Split content by code blocks
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(cleanContent)) !== null) {
    if (match.index > lastIndex) {
      parts.push({
        type: "text",
        value: cleanContent.slice(lastIndex, match.index),
      });
    }

    parts.push({
      type: "code",
      language: match[1] || "text",
      value: match[2].trim(),
    });

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < cleanContent.length) {
    parts.push({
      type: "text",
      value: cleanContent.slice(lastIndex),
    });
  }

  return (
    <div className="markdown-body">
      {parts.map((part, idx) => {
        if (part.type === "code") {
          return <CodeBlock key={idx} language={part.language} code={part.value} />;
        }
        return <FormattedText key={idx} text={part.value} onImageClick={onImageClick} />;
      })}
    </div>
  );
}

function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="code-block-container">
      <div className="code-block-header">
        <span className="code-lang">{language || "code"}</span>
        <button onClick={handleCopy} className="code-copy-btn" type="button">
          {copied ? <Check size={14} className="icon-success" /> : <Copy size={14} />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre className="code-content">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function FormattedText({ text, onImageClick }) {
  const lines = text.split("\n");
  const elements = [];
  let inList = false;
  let listItems = [];

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} className="md-list">
          {listItems.map((item, i) => (
            <li key={i}>{formatInline(item)}</li>
          ))}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  };

  let inTable = false;
  let tableRows = [];

  const flushTable = () => {
    if (tableRows.length > 0) {
      const header = tableRows[0];
      const rows = tableRows.slice(1).filter((r) => !r.every((c) => c.match(/^:?-+:?$/)));
      elements.push(
        <div key={`tbl-${elements.length}`} className="md-table-container custom-scrollbar">
          <table className="md-table">
            <thead>
              <tr>
                {header.map((col, cIdx) => (
                  <th key={cIdx}>{formatInline(col.trim())}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rIdx) => (
                <tr key={rIdx}>
                  {row.map((col, cIdx) => (
                    <td key={cIdx}>{formatInline(col.trim())}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableRows = [];
      inTable = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Markdown Table rows
    if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
      flushList();
      inTable = true;
      const cells = line
        .trim()
        .slice(1, -1)
        .split("|")
        .map((c) => c.trim());
      tableRows.push(cells);
      continue;
    } else {
      flushTable();
    }

    // Bullet list items
    if (line.trim().startsWith("- ") || line.trim().startsWith("* ")) {
      inList = true;
      listItems.push(line.trim().substring(2));
      continue;
    } else {
      flushList();
    }

    // Numbered lists
    const numMatch = line.trim().match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      elements.push(
        <div key={i} className="md-num-item">
          <span className="md-num-index">{numMatch[1]}.</span>
          <span>{formatInline(numMatch[2])}</span>
        </div>
      );
      continue;
    }

    // Markdown Images ![alt](url)
    const imgMatch = line.trim().match(/^!\[(.*?)\]\((.*?)\)/);
    if (imgMatch) {
      const altText = imgMatch[1] || "Document Figure";
      const rawImgUrl = imgMatch[2];
      const resolvedUrl = resolveImageUrl(rawImgUrl);
      elements.push(
        <div key={i} className="md-image-container">
          <img
            src={resolvedUrl}
            alt={altText}
            className="md-rendered-image"
            loading="lazy"
            onClick={() => {
              if (onImageClick) {
                onImageClick(resolvedUrl, altText);
              } else {
                window.open(resolvedUrl, "_blank");
              }
            }}
            onError={(e) => {
              e.target.style.display = "none";
            }}
          />
          {altText && (
            <div className="md-image-caption">
              {altText}
            </div>
          )}
        </div>
      );
      continue;
    }

    // Callout Alerts > [!NOTE], > [!WARNING], > [!TIP]
    const calloutMatch = line.trim().match(/^>\s*\[!(NOTE|WARNING|TIP|IMPORTANT|CAUTION)\]\s*(.*)/i);
    if (calloutMatch) {
      const type = calloutMatch[1].toUpperCase();
      const calloutText = calloutMatch[2];
      const isWarn = type === "WARNING" || type === "CAUTION";
      const isTip = type === "TIP";
      const isImportant = type === "IMPORTANT";
      
      elements.push(
        <div key={i} className={`md-callout md-callout-${type.toLowerCase()}`}>
          <div className="md-callout-header">
            {isWarn ? (
              <ShieldAlert size={15} />
            ) : isTip ? (
              <Lightbulb size={15} />
            ) : isImportant ? (
              <AlertTriangle size={15} />
            ) : (
              <Info size={15} />
            )}
            <span>{type}</span>
          </div>
          {calloutText && <div className="md-callout-body">{formatInline(calloutText)}</div>}
        </div>
      );
      continue;
    }

    // Headers
    if (line.startsWith("### ")) {
      const title = line.substring(4);
      const isWarningHeader = title.includes("⚠️") || title.toLowerCase().includes("missing information");
      if (isWarningHeader) {
        elements.push(
          <div key={i} className="md-missing-info-header">
            <AlertTriangle size={16} className="text-warning" />
            <span>{formatInline(title.replace("⚠️", "").trim())}</span>
          </div>
        );
      } else {
        elements.push(<h4 key={i} className="md-h4">{formatInline(title)}</h4>);
      }
    } else if (line.startsWith("## ")) {
      elements.push(<h3 key={i} className="md-h3">{formatInline(line.substring(3))}</h3>);
    } else if (line.startsWith("# ")) {
      elements.push(<h2 key={i} className="md-h2">{formatInline(line.substring(2))}</h2>);
    } else if (line.startsWith("> ")) {
      elements.push(<blockquote key={i} className="md-quote">{formatInline(line.substring(2))}</blockquote>);
    } else if (!line.trim()) {
      elements.push(<div key={i} className="md-spacer" />);
    } else {
      elements.push(<p key={i} className="md-p">{formatInline(line)}</p>);
    }
  }

  flushList();
  flushTable();

  return <>{elements}</>;
}

function formatInline(str) {
  if (!str) return "";

  const tokens = [];
  let remaining = str;
  let key = 0;

  while (remaining.length > 0) {
    // Citation Badges: [1], [2], [Context 1], [Page 4], [Doc 1]
    const citeMatch = remaining.match(/^\[(\d+|Context\s*\d+|Page\s*\d+|Doc\s*\d+|Ref\s*\d+)\]/i);
    if (citeMatch) {
      tokens.push(
        <span key={key++} className="md-citation-badge" title={`Source Reference ${citeMatch[1]}`}>
          {citeMatch[1]}
        </span>
      );
      remaining = remaining.substring(citeMatch[0].length);
      continue;
    }

    // Inline code `...`
    const codeMatch = remaining.match(/^`([^`]+)`/);
    if (codeMatch) {
      tokens.push(<code key={key++} className="inline-code">{codeMatch[1]}</code>);
      remaining = remaining.substring(codeMatch[0].length);
      continue;
    }

    // Bold **...**
    const boldMatch = remaining.match(/^\*\*([^*]+)\*\*/);
    if (boldMatch) {
      tokens.push(<strong key={key++}>{boldMatch[1]}</strong>);
      remaining = remaining.substring(boldMatch[0].length);
      continue;
    }

    // Italic *...*
    const italicMatch = remaining.match(/^\*([^*]+)\*/);
    if (italicMatch) {
      tokens.push(<em key={key++}>{italicMatch[1]}</em>);
      remaining = remaining.substring(italicMatch[0].length);
      continue;
    }

    // Inline Links [text](url)
    const linkMatch = remaining.match(/^\[([^\]]+)\]\(([^)]+)\)/);
    if (linkMatch) {
      tokens.push(
        <a
          key={key++}
          href={linkMatch[2]}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline hover:opacity-80"
        >
          {linkMatch[1]}
        </a>
      );
      remaining = remaining.substring(linkMatch[0].length);
      continue;
    }

    // Plain text until next token
    const nextSpecial = remaining.search(/[`*\[]/);
    if (nextSpecial === -1) {
      tokens.push(remaining);
      break;
    } else if (nextSpecial === 0) {
      tokens.push(remaining[0]);
      remaining = remaining.substring(1);
    } else {
      tokens.push(remaining.substring(0, nextSpecial));
      remaining = remaining.substring(nextSpecial);
    }
  }

  return tokens;
}
