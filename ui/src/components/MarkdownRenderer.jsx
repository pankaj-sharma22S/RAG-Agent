import { useState } from "react";
import { Copy, Check } from "lucide-react";

export default function MarkdownRenderer({ content }) {
  if (!content) return null;

  // Split content by code blocks
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push({
        type: "text",
        value: content.slice(lastIndex, match.index),
      });
    }

    parts.push({
      type: "code",
      language: match[1] || "text",
      value: match[2].trim(),
    });

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push({
      type: "text",
      value: content.slice(lastIndex),
    });
  }

  return (
    <div className="markdown-body">
      {parts.map((part, idx) => {
        if (part.type === "code") {
          return <CodeBlock key={idx} language={part.language} code={part.value} />;
        }
        return <FormattedText key={idx} text={part.value} />;
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

function FormattedText({ text }) {
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

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

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

    // Headers
    if (line.startsWith("### ")) {
      elements.push(<h4 key={i} className="md-h4">{formatInline(line.substring(4))}</h4>);
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

  return <>{elements}</>;
}

function formatInline(str) {
  if (!str) return "";

  const tokens = [];
  let remaining = str;
  let key = 0;

  while (remaining.length > 0) {
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

    // Plain text until next token
    const nextSpecial = remaining.search(/[`*]/);
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
