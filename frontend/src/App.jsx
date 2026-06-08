import { useState, useRef, useCallback, useEffect } from 'react';

const METRICS = [
  { label: 'Accuracy', value: '77.3%' },
  { label: 'AUC-ROC', value: '0.845' },
  { label: 'F1 Score', value: '0.73' },
  { label: 'Precision', value: '0.74' },
];

const API_URL = 'http://localhost:8000/api/v1/detect';

/* ─── Upload Icon ─── */

function UploadIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#111" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

/* ─── Dropzone ─── */

function Dropzone({ onFile }) {
  const [over, setOver] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setOver(false);
      const f = e.dataTransfer.files[0];
      if (f) onFile(f);
    },
    [onFile]
  );

  return (
    <div
      className={`dropzone anim-fade-up anim-delay-3${over ? ' dragover' : ''}`}
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={handleDrop}
    >
      <div className="dropzone-icon-wrap">
        <span className="dropzone-icon"><UploadIcon /></span>
      </div>
      <div className="dropzone-text">
        {over ? 'Drop image here' : 'Upload an image'}
      </div>
      <div className="dropzone-hint">JPG, JPEG or PNG &nbsp;·&nbsp; 12 KB – 10 MB</div>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png"
        onChange={(e) => {
          const f = e.target.files[0];
          if (f) onFile(f);
        }}
      />
    </div>
  );
}

/* ─── Metrics Bar ─── */

function MetricsBar() {
  return (
    <div className="metrics-bar anim-fade-up anim-delay-1">
      {METRICS.map((m, i) => (
        <div key={m.label} className="metric">
          <div className="metric-value">{m.value}</div>
          <div className="metric-label">{m.label}</div>
        </div>
      ))}
    </div>
  );
}

/* ─── Result Card ─── */

function ResultCard({ result }) {
  if (!result) return null;
  const isForged = result.verdict === 'FORGED';

  return (
    <div className="result-card anim-scale-in">
      <div className="result-header">
        <div className="result-header-left">
          <span className={`result-dot ${isForged ? 'forged' : 'authentic'}`} />
          <span className="result-verdict-label">Evaluation Result</span>
        </div>
        <span className={`result-verdict ${isForged ? 'forged' : 'authentic'}`}>
          {result.verdict}
        </span>
      </div>

      <div className="result-stats">
        <div className="stat-box">
          <div className="stat-label">Confidence</div>
          <div className="stat-value">{Number(result.confidence).toFixed(2)}%</div>
          <div className="confidence-bar-wrap">
            <div
              className={`confidence-bar ${isForged ? 'forged' : 'authentic'}`}
              style={{ width: `${Math.min(result.confidence, 100)}%` }}
            />
          </div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Processing Time</div>
          <div className="stat-value">{result.processing_time_ms.toFixed(1)} ms</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Forgery Type</div>
          <div className="stat-value">
            {isForged ? 'Splicing / Copy-Move' : '—'}
          </div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Model Threshold</div>
          <div className="stat-value">0.45</div>
        </div>
      </div>
    </div>
  );
}

/* ─── About Project ─── */

const TILES = [
  { icon: '🧬', title: 'Dual-Branch CNN', metric: 'Accuracy', value: '77.3%', wide: true },
  { icon: '🎯', title: 'Precision', metric: 'Precision', value: '0.74', wide: false },
  { icon: '🔍', title: 'Splicing & Copy-Move', metric: 'F1 Score', value: '0.73', wide: false },
  { icon: '📊', title: 'ROC Analysis', metric: 'AUC-ROC', value: '0.845', wide: true },
  { icon: '⚡', title: 'Inference Speed', metric: 'Per Image', value: '<100ms', wide: true },
  { icon: '📋', title: 'Training Data', metric: 'CASIA v2', value: '12K+ images', wide: false },
];

function AboutProject() {
  return (
    <section className="about-project anim-fade-up">
      <div className="about-decor">
        <span className="about-line" />
        <span className="about-decor-dot" />
        <span className="about-line" />
      </div>
      <h2 className="about-heading">About DeepSight</h2>
      <p className="about-text">
        DeepSight uses a <strong>dual-branch CNN</strong> to detect image
        forgeries. The <strong>RGB branch</strong> extracts visual features
        via EfficientNet-B0, while the <strong>noise branch</strong> analyses
        30 SRM filter responses and ELA residuals to reveal tampering invisible
        to the naked eye. Both streams are fused and classified — optimised for{' '}
        <strong>splicing</strong> and <strong>copy-move</strong> detection.
      </p>
      <div className="about-tiles">
        {TILES.map((t) => (
          <div
            key={t.title}
            className={`about-tile${t.wide ? ' about-tile-wide' : ''}`}
          >
            <span className="about-tile-icon">{t.icon}</span>
            <div className="about-tile-content">
              <div className="about-tile-title">{t.title}</div>
              <div className="about-tile-metric">{t.value}</div>
              <div className="about-tile-label">{t.metric}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ─── Footer ─── */

function Footer({ onNavigate }) {
  return (
    <footer className="footer anim-fade-up">
      <div className="footer-top">
        <div className="footer-col">
          <div className="footer-brand">
            <span className="footer-brand-dot" />
            <span className="footer-brand-name">DeepSight</span>
          </div>
          <span className="footer-text">Image Forgery Detection</span>
          <span className="footer-text">v1.0 &mdash; Academic Project</span>
        </div>
        <div className="footer-col">
          <div className="footer-col-title">Explore</div>
          <button className="footer-link" onClick={() => onNavigate('devs')}>
            Meet the team
          </button>
          <button className="footer-link" onClick={() => onNavigate('how')}>
            How it works
          </button>
        </div>
        <div className="footer-col">
          <div className="footer-col-title">Resources</div>
          <a
            href="https://github.com/hamzasterical/deepsight"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-link"
          >
            GitHub Repository
          </a>
          <a
            href="https://www.kaggle.com/datasets/sophatvathana/casia-dataset"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-link"
          >
            CASIA v2 Dataset
          </a>
        </div>
      </div>
      <div className="footer-bottom">
        <div className="footer-bottom-left">
          <span className="footer-dot" />
          <span>&copy; 2026 DeepSight &mdash; Built with PyTorch &amp; FastAPI</span>
        </div>
        <span>Splicing &amp; Copy-Move</span>
      </div>
    </footer>
  );
}

/* ─── Devs Page ─── */

const DEVS = [
  {
    name: 'Hamza',
    role: 'AI Engineer & ML Developer',
    desc: 'Model architecture, dual-branch CNN design, training pipeline, and API development.',
  },
  {
    name: 'Umer Farooq',
    role: 'Frontend Developer',
    desc: 'Frontend design, visual identity, React implementation, and user experience.',
  },
  {
    name: 'GM Khizar',
    role: 'Documentation & Data Preparator & Evaluator',
    desc: 'Dataset preparation, augmentation pipeline, model evaluation, and project documentation.',
  },
];

function DevsPage({ onNavigate }) {
  return (
    <div className="devs-page">
      <div className="devs-header anim-fade-up">
        <button className="back-btn" onClick={() => onNavigate('home')}>
          &larr; Back
        </button>
      </div>

      <h2 className="devs-heading anim-fade-up anim-delay-1">
        Meet the <span className="green">Team</span>
      </h2>
      <p className="devs-sub anim-fade-up anim-delay-2">
        Built by three developers passionate about forensic image analysis.
      </p>

      <div className="devs-grid">
        {DEVS.map((dev, i) => (
          <div
            key={dev.name}
            className={`dev-card anim-fade-up anim-delay-${i + 3}`}
          >
            <div className="dev-avatar">{dev.name[0]}</div>
            <div className="dev-name">{dev.name}</div>
            <div className="dev-role">{dev.role}</div>
            <div className="dev-desc">{dev.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── How It Works Page ─── */

/* =================================================================
   IMAGE GENERATION PROMPTS FOR GEMINI
   =================================================================

   Generate these images with Gemini and place them in:
   frontend/public/images/

   PROMPT 1 — Pipeline Overview
   "A sleek, modern horizontal flowchart illustration showing 5 steps
   of an image forgery detection pipeline: Upload Image, Preprocess,
   CNN Analysis, Classify, Verdict. Connected by arrows. Dark
   background with amber/gold accent colors. Tech/AI aesthetic.
   Minimal flat design style. 1200x300 px."

   PROMPT 2 — Forgery Types Comparison
   "Side by side comparison showing two types of image forgery.
   Left side 'Splicing': an image with a person cut from one photo
   pasted into another scene. Right side 'Copy-Move': an image with
   a duplicated object (like a car) moved to cover another area.
   Both with subtle dashed outlines highlighting tampered regions.
   Clean forensic/investigative aesthetic. 800x400 px."

   PROMPT 3 — Dual-Branch CNN Architecture
   "A vertical block diagram / architecture diagram of a dual-branch
   convolutional neural network for image forensics. Top: Input Image.
   Splits into two branches: Left branch labeled 'RGB Branch' with
   'EfficientNet-B0' and 'Visual Features'. Right branch labeled
   'Noise Branch' with '30 SRM Filters' and 'ELA Residuals'. Both
   merge into 'Feature Fusion' then 'Classifier' then output
   'Forgery / Authentic'. Clean technical diagram style, dark
   background with warm amber tones. 800x500 px."

   PROMPT 4 — Detection Heatmap Result
   "A split view showing a forged image on the left and its
   deep learning detection heatmap on the right. The heatmap has
   warm red/orange highlights over the tampered region. Forensic
   analysis aesthetic with subtle grid overlay. 800x400 px."

   PROMPT 5 — SRM Filter Visualization
   "A 3x3 grid of grayscale SRM (Spatial Rich Model) filter
   visualizations showing noise patterns in images. Some filters
   highlight edges, others textures, others noise residuals.
   Scientific/forensic aesthetic, monochrome with subtle warm
   accent on one filter. 600x600 px."
   ================================================================= */

function CNNArchDiagram() {
  return (
    <svg viewBox="0 0 820 520" className="arch-svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" fill="#bbb" />
        </marker>
      </defs>

      <rect x="350" y="6" width="120" height="44" rx="8" fill="#EFEFED" stroke="#8B6F47" strokeWidth="2" />
      <text x="410" y="34" textAnchor="middle" fill="#111" fontSize="13" fontWeight="700">Input Image</text>

      <line x1="410" y1="50" x2="410" y2="76" stroke="#bbb" strokeWidth="2" />
      <line x1="180" y1="76" x2="640" y2="76" stroke="#bbb" strokeWidth="2" />
      <line x1="180" y1="76" x2="180" y2="100" stroke="#bbb" strokeWidth="2" markerEnd="url(#arrow)" />
      <line x1="640" y1="76" x2="640" y2="100" stroke="#bbb" strokeWidth="2" markerEnd="url(#arrow)" />

      <rect x="80" y="104" width="200" height="42" rx="8" fill="#4A3228" />
      <text x="180" y="130" textAnchor="middle" fill="#fff" fontSize="13" fontWeight="600">RGB Branch</text>

      <rect x="540" y="104" width="200" height="42" rx="8" fill="#4A3228" />
      <text x="640" y="130" textAnchor="middle" fill="#fff" fontSize="13" fontWeight="600">Noise Branch</text>

      <rect x="90" y="164" width="180" height="34" rx="6" fill="#EFEFED" stroke="#ddd" strokeWidth="1" />
      <text x="180" y="185" textAnchor="middle" fill="#555" fontSize="11">EfficientNet-B0</text>

      <rect x="90" y="210" width="180" height="34" rx="6" fill="#EFEFED" stroke="#ddd" strokeWidth="1" />
      <text x="180" y="231" textAnchor="middle" fill="#555" fontSize="11">Visual Features</text>

      <line x1="180" y1="146" x2="180" y2="164" stroke="#bbb" strokeWidth="1.5" />
      <line x1="180" y1="198" x2="180" y2="210" stroke="#bbb" strokeWidth="1.5" />

      <rect x="550" y="164" width="180" height="34" rx="6" fill="#EFEFED" stroke="#ddd" strokeWidth="1" />
      <text x="640" y="185" textAnchor="middle" fill="#555" fontSize="11">30 SRM Filters</text>

      <rect x="550" y="210" width="180" height="34" rx="6" fill="#EFEFED" stroke="#ddd" strokeWidth="1" />
      <text x="640" y="231" textAnchor="middle" fill="#555" fontSize="11">ELA Residuals</text>

      <line x1="640" y1="146" x2="640" y2="164" stroke="#bbb" strokeWidth="1.5" />
      <line x1="640" y1="198" x2="640" y2="210" stroke="#bbb" strokeWidth="1.5" />

      <line x1="180" y1="244" x2="180" y2="290" stroke="#bbb" strokeWidth="2" />
      <line x1="640" y1="244" x2="640" y2="290" stroke="#bbb" strokeWidth="2" />
      <line x1="180" y1="290" x2="640" y2="290" stroke="#bbb" strokeWidth="2" />
      <line x1="410" y1="290" x2="410" y2="320" stroke="#bbb" strokeWidth="2" markerEnd="url(#arrow)" />

      <rect x="315" y="324" width="190" height="42" rx="8" fill="#8B6F47" />
      <text x="410" y="349" textAnchor="middle" fill="#fff" fontSize="13" fontWeight="600">Feature Fusion</text>

      <line x1="410" y1="366" x2="410" y2="396" stroke="#bbb" strokeWidth="2" markerEnd="url(#arrow)" />

      <rect x="315" y="400" width="190" height="42" rx="8" fill="#4A3228" />
      <text x="410" y="425" textAnchor="middle" fill="#fff" fontSize="13" fontWeight="600">Classifier</text>

      <line x1="410" y1="442" x2="410" y2="465" stroke="#bbb" strokeWidth="2" markerEnd="url(#arrow)" />
      <rect x="360" y="469" width="100" height="20" rx="4" fill="#EFEFED" stroke="#ddd" strokeWidth="1" />
      <text x="410" y="483" textAnchor="middle" fill="#111" fontSize="11" fontWeight="600">Forgery / Authentic</text>
    </svg>
  );
}

function ForgedImageSVG({ label, description }) {
  return (
    <div className="forgery-visual">
      <svg viewBox="0 0 340 220" className="forgery-svg">
        <rect width="340" height="220" rx="10" fill="#EFEFED" />
        <rect x="20" y="16" width="300" height="160" rx="6" fill="#fff" stroke="#ddd" strokeWidth="1" />
        <rect x="50" y="40" width="100" height="80" rx="4" fill="#e8e4e0" />
        <rect x="130" y="90" width="80" height="60" rx="4" fill="#d5cfc8" />
        <rect x="190" y="36" width="90" height="70" rx="4" fill="#ddd8d2" stroke="#8B6F47" strokeWidth="2" strokeDasharray="5,4" />
        <rect x="190" y="36" width="90" height="70" rx="4" fill="rgba(139,111,71,0.15)" />
        <text x="235" y="78" textAnchor="middle" fill="#8B6F47" fontSize="9" fontWeight="600">TAMPER</text>
        <rect x="20" y="186" width="300" height="24" rx="4" fill="#4A3228" />
        <text x="170" y="202" textAnchor="middle" fill="#fff" fontSize="10" fontWeight="600">{label}</text>
      </svg>
      <div className="forgery-desc">{description}</div>
    </div>
  );
}

function HowItWorksPage({ onNavigate }) {
  return (
    <div className="how-page">
      <div className="how-header anim-fade-up">
        <button className="back-btn" onClick={() => onNavigate('home')}>
          &larr; Back
        </button>
      </div>

      <div className="how-content">
        <h1 className="how-title anim-fade-up anim-delay-1">
          How DeepSight Works
        </h1>
        <p className="how-subtitle anim-fade-up anim-delay-2">
          A dual-branch CNN that detects image forgeries by analysing visual
          features and noise-level inconsistencies
        </p>

        {/* ─── Section 1: Pipeline —────────────── */}
        <section className="how-section anim-fade-up anim-delay-3">
          <h2 className="how-section-title">Detection Pipeline</h2>
          <div className="pipeline">
            <div className="pipe-step">
              <div className="pipe-icon"><UploadIcon /></div>
              <div className="pipe-label">Upload</div>
            </div>
            <div className="pipe-arrow">&rarr;</div>
            <div className="pipe-step">
              <div className="pipe-icon" style={{ fontSize: 20 }}>⚙️</div>
              <div className="pipe-label">Preprocess</div>
            </div>
            <div className="pipe-arrow">&rarr;</div>
            <div className="pipe-step">
              <div className="pipe-icon" style={{ fontSize: 20 }}>🧠</div>
              <div className="pipe-label">CNN</div>
            </div>
            <div className="pipe-arrow">&rarr;</div>
            <div className="pipe-step">
              <div className="pipe-icon" style={{ fontSize: 20 }}>📊</div>
              <div className="pipe-label">Classify</div>
            </div>
            <div className="pipe-arrow">&rarr;</div>
            <div className="pipe-step">
              <div className="pipe-icon" style={{ fontSize: 20 }}>✅</div>
              <div className="pipe-label">Verdict</div>
            </div>
          </div>
          <div className="pipeline-img-placeholder" data-prompt="1" />
        </section>

        {/* ─── Section 2: Forgery Types —───────── */}
        <section className="how-section anim-fade-up anim-delay-4">
          <h2 className="how-section-title">What We Detect</h2>
          <div className="forgery-types">
            <ForgedImageSVG
              label="SPLICING"
              description="A region from one image is copied and pasted into another"
            />
            <ForgedImageSVG
              label="COPY-MOVE"
              description="A part of the image is duplicated and moved within the same image"
            />
          </div>
        </section>

        {/* ─── Section 3: Architecture —────────── */}
        <section className="how-section anim-fade-up anim-delay-5">
          <h2 className="how-section-title">Dual-Branch CNN Architecture</h2>
          <div className="arch-wrap">
            <CNNArchDiagram />
          </div>
        </section>

        {/* ─── Section 4: How Detection Works —─── */}
        <section className="how-section anim-fade-up anim-delay-1">
          <h2 className="how-section-title">Detection Mechanics</h2>
          <div className="mechanics-grid">
            <div className="mech-card">
              <div className="mech-icon">🔬</div>
              <div className="mech-bar" style={{ '--pct': '85%' }} />
              <div className="mech-label">Noise Inconsistency</div>
            </div>
            <div className="mech-card">
              <div className="mech-icon">🧩</div>
              <div className="mech-bar" style={{ '--pct': '78%' }} />
              <div className="mech-label">Boundary Artifacts</div>
            </div>
            <div className="mech-card">
              <div className="mech-icon">🎨</div>
              <div className="mech-bar" style={{ '--pct': '72%' }} />
              <div className="mech-label">Texture Anomalies</div>
            </div>
            <div className="mech-card">
              <div className="mech-icon">⚡</div>
              <div className="mech-bar" style={{ '--pct': '90%' }} />
              <div className="mech-label">Fusion &amp; Decision</div>
            </div>
          </div>
        </section>

        {/* ─── Section 5: Output Examples —─────── */}
        <section className="how-section anim-fade-up anim-delay-2">
          <h2 className="how-section-title">Detection Result</h2>
          <div className="detection-demo">
            <div className="demo-half">
              <div className="demo-placeholder">
                <svg viewBox="0 0 280 200" className="demo-svg">
                  <rect width="280" height="200" rx="10" fill="#EFEFED" />
                  <rect x="60" y="30" width="160" height="100" rx="6" fill="#fff" stroke="#ddd" strokeWidth="1" />
                  <rect x="100" y="50" width="80" height="50" rx="4" fill="#e0dbd4" stroke="#8B6F47" strokeWidth="2" strokeDasharray="5,4" />
                  <text x="140" y="82" textAnchor="middle" fill="#8B6F47" fontSize="11" fontWeight="700">?</text>
                  <text x="140" y="190" textAnchor="middle" fill="#888" fontSize="10">Uploaded Image</text>
                </svg>
              </div>
            </div>
            <div className="demo-arrow">&#8594;</div>
            <div className="demo-half">
              <div className="demo-placeholder">
                <svg viewBox="0 0 340 200" className="demo-svg">
                  <rect width="340" height="200" rx="10" fill="#EFEFED" />
                  <rect x="20" y="16" width="300" height="120" rx="6" fill="#fff" stroke="#ddd" strokeWidth="1" />
                  <rect x="40" y="36" width="260" height="80" rx="6" fill="#f5f2ee" />
                  <rect x="100" y="46" width="60" height="40" rx="4" fill="rgba(139,111,71,0.25)" stroke="#8B6F47" strokeWidth="2" />
                  <rect x="100" y="46" width="60" height="40" rx="4" fill="rgba(139,111,71,0.15)" />
                  <text x="130" y="70" textAnchor="middle" fill="#8B6F47" fontSize="9" fontWeight="600">FORGERY</text>
                  <line x1="100" y1="86" x2="160" y2="86" stroke="#8B6F47" strokeWidth="1.5" strokeDasharray="3,3" />
                  <text x="170" y="186" textAnchor="middle" fill="#111" fontSize="11" fontWeight="700">Confidence: 87.4%</text>
                  <text x="170" y="155" textAnchor="middle" fill="#888" fontSize="10">Heatmap Overlay</text>
                </svg>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

/* ─── Home Page ─── */

function HomePage({ onFile, preview, loading, result, error, onNavigate }) {
  return (
    <>
      <span className="bg-glow" />
      <span className="bg-glow-2" />

      <header className="header anim-fade-up">
        <div className="header-left">
          <span className="brand-dot" />
          <span className="brand-name">DeepSight</span>
        </div>
        <span className="header-label">
          CopyMove + splice image forgery detector
        </span>
      </header>

      <MetricsBar />

      <div className="upload-section">
        <Dropzone onFile={onFile} />

        {preview && (
          <div className="preview-section anim-fade-in">
            <div className="preview-container">
              <img src={preview} alt="Uploaded preview" />
            </div>
          </div>
        )}
      </div>

      {loading && (
        <div className="spinner-wrap anim-fade-in">
          <div className="spinner" />
        </div>
      )}

      {error && <div className="error-box anim-fade-in">{error}</div>}

      {result && <ResultCard result={result} />}

      <AboutProject />

      <Footer onNavigate={onNavigate} />
    </>
  );
}

/* ─── App ─── */

export default function App() {
  const [page, setPage] = useState('home');
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFile = useCallback((file) => {
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    setLoading(true);

    const form = new FormData();
    form.append('file', file);

    fetch(API_URL, { method: 'POST', body: form })
      .then(async (r) => {
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `Server error (${r.status})`);
        return data;
      })
      .then((data) => setResult(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (page === 'devs') {
    return <DevsPage onNavigate={setPage} />;
  }

  if (page === 'how') {
    return <HowItWorksPage onNavigate={setPage} />;
  }

  return (
    <HomePage
      onFile={handleFile}
      preview={preview}
      loading={loading}
      result={result}
      error={error}
      onNavigate={setPage}
    />
  );
}
