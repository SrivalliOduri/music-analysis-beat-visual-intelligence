from flask import Flask, request, render_template_string, send_from_directory, redirect
from werkzeug.utils import secure_filename
from pathlib import Path
from functools import lru_cache
import uuid
import librosa
import numpy as np
import pandas as pd
import joblib

try:
    import opensmile
except ImportError:
    opensmile = None

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "ml" / "emotion_model.joblib"
COLUMNS_PATH = BASE_DIR / "dataset" / "columns.csv"
REFERENCE_PATH = BASE_DIR / "dataset" / "deam_song_level_features.csv"

UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_FILE_BYTES = 20 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = 2 * MAX_FILE_BYTES

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SONIQ | Music Analysis</title>
<style>
*{box-sizing:border-box}
:root{
--panel:rgba(24,43,41,.88);--text:#f2f5ec;
--muted:#bac9bd;--accent:#b6d8c0;
--border:rgba(216,237,218,.18)
}
html{scroll-behavior:smooth}
body{
margin:0;padding:25px 15px;min-height:100vh;color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
background:linear-gradient(120deg,rgba(9,25,24,.9),rgba(17,39,31,.85)),
url("https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=2000&q=85") center/cover fixed
}
.container{
max-width:1100px;margin:auto;padding:28px;
background:rgba(15,29,28,.85);
border:1px solid var(--border);border-radius:24px;
backdrop-filter:blur(18px)
}
nav{
position:sticky;top:0;z-index:1000;
display:flex;justify-content:space-between;align-items:center;
gap:15px;flex-wrap:wrap;padding:12px 0 18px;
background:rgba(15,29,28,.96);
backdrop-filter:blur(16px);
border-bottom:1px solid var(--border)
}
.brand{color:#d9ead9;font-size:22px;font-weight:800;letter-spacing:2px}
.nav-links{display:flex;gap:6px;flex-wrap:wrap}
.nav-link{color:var(--muted);text-decoration:none;padding:9px;border-radius:9px}
.nav-link:hover,.active{background:#ffffff14;color:#e6f3e7}
.hero{padding:28px 0 15px}
.eyebrow{color:var(--accent);font-size:11px;letter-spacing:3px;font-weight:700}
h1{font-size:clamp(32px,5vw,48px);letter-spacing:-1px;color:#d9ead9;margin:8px 0}
h2,h3{color:#d6e9d8}
p{color:var(--muted);line-height:1.7}
input,button,select{margin:8px 3px;padding:13px 16px;border-radius:12px;font:inherit}
input[type=file]{max-width:100%;color:var(--muted);background:#ffffff09;border:1px solid var(--border)}
select{color:var(--text);background:#ffffff09;border:1px solid var(--border);max-width:100%}
option{color:#142621}
button{
background:linear-gradient(135deg,#b6d8c0,#7fae9a);
color:#142621;border:0;cursor:pointer;font-weight:700
}
button:hover{filter:brightness(1.08)}
button:disabled{opacity:.7}
audio{width:100%;max-width:600px;margin:15px 0}
.panel{
scroll-margin-top:100px;
background:var(--panel);border:1px solid var(--border);
border-radius:20px;padding:23px;margin-top:22px
}
.bpm{font-size:clamp(38px,6vw,54px);font-weight:750;color:#c3e7ce}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:13px;margin-top:18px}
.card{padding:17px;border-radius:15px;background:#ffffff08;border:1px solid var(--border)}
.card strong{display:block;color:#d6e9d8;margin-bottom:8px}
canvas{display:block;width:100%;height:300px;background:#081616;border:1px solid var(--border);border-radius:16px}
#timeline{height:120px;cursor:pointer}
.timestamps{
max-height:220px;overflow:auto;overflow-wrap:anywhere;
white-space:pre-line;color:var(--muted);line-height:1.8
}
button.event-item{
display:block;width:100%;margin:5px 0;padding:10px;text-align:left;
background:rgba(255,255,255,0.06);color:var(--text);
border:1px solid var(--border);border-radius:8px;cursor:pointer;font-weight:400
}
button.event-item:hover{background:rgba(182,216,192,0.18);filter:none}
table{width:100%;border-collapse:collapse;margin-top:14px}
th,td{padding:10px;text-align:left;border-bottom:1px solid var(--border);color:var(--muted)}
th{color:#d6e9d8}
li{margin:10px 0;line-height:1.6;color:var(--muted)}
small,footer{color:var(--muted)}
footer{margin-top:30px;padding-top:15px;border-top:1px solid var(--border);font-size:13px}
@media(max-width:600px){
body{padding:10px}
.container{padding:17px 13px}
.panel{padding:16px}
.nav-links{width:100%}
}
</style>
</head>
<body>
<div class="container">
<nav>
<a class="brand" href="#home" style="text-decoration:none">🌿 SONIQ</a>
<div class="nav-links">
<a class="nav-link active" href="#home">Home</a>
<a class="nav-link" href="#analysis">Analysis</a>
<a class="nav-link" href="#visualizer-section">Visualizer</a>
<a class="nav-link" href="#compare">Compare</a>
<a class="nav-link" href="#history">History</a>
<a class="nav-link" href="#insights">Insights</a>
<a class="nav-link" href="#about">About</a>
</div>
</nav>

<section id="home" class="hero">
<p class="eyebrow">MUSIC · NATURE · INTELLIGENCE</p>
<h1>Your Music, Decoded.</h1>
<p>Find the rhythm. Feel the atmosphere. Explore every sound.</p>
</section>

<section id="analysis" class="panel">
<h2>Explore Your Sound</h2>
<p>Upload an MP3 or WAV file to analyse its tempo, beats, acoustic features and predicted mood.</p>
<form method="POST" action="/" enctype="multipart/form-data">
<input type="file" name="audio" accept=".mp3,.wav" required>
<br><button type="submit">Upload &amp; Analyse</button>
</form>
{% if message %}<p>{{ message }}</p>{% endif %}
</section>

{% if audio_url %}
<section class="panel">
<h2>Your Song</h2>
<p><strong>File:</strong> {{ song_name }}</p>
<audio id="player" controls>
<source src="{{ audio_url }}" type="{{ audio_type }}">
</audio>

<h2>Music Analysis Results</h2>
<button id="saveAnalysis" type="button">Save This Analysis</button>
<p id="saveStatus" aria-live="polite"></p>
<p>Estimated Tempo</p>
<div class="bpm">{{ results.bpm }} BPM</div>

<div class="grid">
<div class="card"><strong>Duration</strong>{{ results.duration }} seconds</div>
<div class="card"><strong>Detected Beats</strong>{{ results.beat_count }}</div>
<div class="card"><strong>Average RMS Energy</strong>{{ results.rms_energy }}</div>
<div class="card"><strong>Spectral Centroid</strong>{{ results.spectral_centroid_hz }} Hz</div>
<div class="card"><strong>Zero-Crossing Rate</strong>{{ results.zero_crossing_rate }}</div>
<div class="card"><strong>Energy Level</strong>{{ results.energy_description }}</div>
<div class="card"><strong>Spectral Brightness</strong>{{ results.brightness_description }}</div>
</div>

<h2>AI Music Emotion Analysis</h2>
{% if results.emotion %}
<div class="grid">
<div class="card"><strong>Predicted Valence</strong>{{ results.emotion.valence }} / 9</div>
<div class="card"><strong>Predicted Arousal</strong>{{ results.emotion.arousal }} / 9</div>
<div class="card"><strong>Estimated Audio Profile</strong>{{ results.emotion.mood }}</div>
<div class="card">
<strong>Feature Compatibility Check</strong>
{{ results.emotion.reliability }}<br>
<small>{{ results.emotion.range_outside }} of {{ results.emotion.range_checked }} features outside the DEAM 1st–99th percentile range; {{ results.emotion.imputed }} missing values imputed.</small>
</div>
</div>
<p><small>These are model estimates, not definitive emotional labels. The compatibility check only compares feature ranges with the training data; it does not prove the extraction pipeline matches DEAM's, and it is not a measured accuracy.</small></p>
{% else %}
<p>Emotion prediction unavailable: {{ results.emotion_error }}</p>
{% endif %}

<h3>Beat Timestamps (seconds)</h3>
<div class="timestamps">{{ results.timestamps }}</div>

<h3 id="insights">Music Improvement Suggestions</h3>
<ul>{% for s in results.suggestions %}<li>{{ s }}</li>{% endfor %}</ul>

<h3 id="pathway">Music Development Pathway</h3>
<p>Use these analysis-based suggestions to explore possible improvements to your arrangement.</p>
<ol>
{% for step in results.development_pathway %}
<li><strong>{{ step.title }}</strong><br>{{ step.detail }}</li>
{% endfor %}
</ol>
<p><small>To check a revision with numbers, use <a class="nav-link" href="#compare" style="display:inline;padding:2px 6px">Compare Two Versions</a>.</small></p>
</section>

<section id="visualizer-section" class="panel">
<h2>Live Beat-Responsive Visualizer</h2>
<p>Enable live visuals and play your song to animate the patterns with its sound and detected beats.</p>
<canvas id="visualizer"></canvas>
<p><small>Track visuals use the audio frequencies and the beat timestamps detected by the analysis. Microphone visuals respond to live frequencies and intensity.</small></p>
<button id="visualButton" type="button">Enable Track Visuals</button>
<button id="micButton" type="button">Start Microphone Visuals</button>
<p id="visualStatus">Enable visuals, then play your song.</p>

<h3>Signal Timeline (click to jump)</h3>
<canvas id="timeline"></canvas>
<p><small>Line: smoothed loudness (RMS, dB) computed on the server. Orange / blue bands: high / low-intensity regions relative to the surrounding ~4 seconds. Faint ticks: detected beats.</small></p>
<p id="agreement"></p>
<h3>Reference Events (from the signal envelope)</h3>
<div id="refEvents" class="timestamps">No reference events found.</div>
<h3>Live Audio Event Timeline</h3>
<p id="eventStatus">Play your song to detect intensity changes.</p>
<div id="eventLog" class="timestamps">No events detected yet.</div>
</section>
<script id="music-analysis-data" type="application/json">{{ results | tojson }}</script>
{% endif %}

<section id="compare" class="panel">
<h2>Compare Two Versions</h2>
<p>Compare two revisions of the same song or compare two different tracks. SONIQ reports measured differences; it does not decide which version or track is better. The app does not automatically verify that two uploads are the same song.</p>
<form method="POST" action="/compare" enctype="multipart/form-data">
<label for="compareMode">Comparison type</label><br>
<select name="compare_mode" id="compareMode">
<option value="revision" {% if compare_mode == "revision" %}selected{% endif %}>Two versions of the same song (revision check)</option>
<option value="tracks" {% if compare_mode == "tracks" %}selected{% endif %}>Two different tracks (descriptive comparison)</option>
</select><br>
<label for="audioA">Version / track A</label><br>
<input id="audioA" type="file" name="audio_a" accept=".mp3,.wav" required><br>
<label for="audioB">Version / track B</label><br>
<input id="audioB" type="file" name="audio_b" accept=".mp3,.wav" required><br>
<div id="intentWrap">
<label for="intentSelect">Intended revision change</label><br>
<select name="intent" id="intentSelect">
{% for key,label in intents %}<option value="{{ key }}" {% if selected_intent == key %}selected{% endif %}>{{ label }}</option>{% endfor %}
</select>
</div>
<button type="submit">Compare</button>
</form>
{% if compare_message %}<p>{{ compare_message }}</p>{% endif %}
{% if comparison %}
<table>
<tr><th>Measurement</th><th>{{ comparison.a_name }}</th><th>{{ comparison.b_name }}</th><th>Change (B − A)</th></tr>
{% for row in comparison.rows %}
<tr><td>{{ row.label }}</td><td>{{ row.a }}</td><td>{{ row.b }}</td><td>{{ row.change }}</td></tr>
{% endfor %}
</table>
<p><strong>{{ comparison.summary_label }}:</strong> {{ comparison.check }}</p>
<p><small>{{ comparison.note }}</small></p>
{% endif %}
</section>

<section id="history" class="panel">
<h2>Saved Analysis History</h2>
<p>Save analyses in this browser to revisit their key measurements. History is stored locally in this browser and is not uploaded to the server.</p>
<button id="clearHistory" type="button">Clear History</button>
<div id="historyList" class="timestamps">No saved analyses yet.</div>
</section>

<section id="about" class="panel">
<h2>About SONIQ</h2>
<p>SONIQ explores music through rhythm detection, acoustic feature analysis, emotion estimation and beat-responsive visualisation.</p>
<p>This prototype uses Librosa for audio analysis, openSMILE for audio descriptors and a trained regression model for valence and arousal prediction.</p>
</section>
<footer>SONIQ · Music Analysis &amp; Beat-Responsive Visual Intelligence</footer>
</div>

<script>

(function () {
  const historyKey = "soniq-analysis-history-v2";

  const saveButton = document.getElementById("saveAnalysis");
  const saveStatus = document.getElementById("saveStatus");
  const historyList = document.getElementById("historyList");
  const clearButton = document.getElementById("clearHistory");

  const resultsPanel =
    document.getElementById("analysis") ||
    document.querySelector(".analysis-results");

  const player = document.getElementById("player");

  function readHistory() {
    try {
      const items = JSON.parse(
        localStorage.getItem(historyKey) || "[]"
      );
      return Array.isArray(items) ? items : [];
    } catch (e) {
      return [];
    }
  }

  function renderHistory() {
    if (!historyList) return;

    const items = readHistory();
    historyList.replaceChildren();

    if (!items.length) {
      historyList.textContent = "No saved analyses yet.";
      return;
    }

    items.forEach((entry, index) => {
      const wrap = document.createElement("div");
      wrap.style.cssText =
        "padding:12px 0;border-bottom:1px solid rgba(216,237,218,.14)";

      const title = document.createElement("strong");
      title.textContent =
        (entry.name || "Audio") + " — " + (entry.savedAt || "");

      const details = document.createElement("p");
      details.textContent =
        `${entry.bpm} BPM · ${entry.duration}s · ` +
        `${entry.beatCount} beats · Energy ${entry.rms}`;

      const viewButton = document.createElement("button");
      viewButton.type = "button";
      viewButton.textContent = "View Analysis";

      viewButton.addEventListener("click", () => {
        let modal = document.getElementById("historyViewModal");

        if (!modal) {
          modal = document.createElement("div");
          modal.id = "historyViewModal";
          modal.style.cssText = `
            position:fixed; inset:0; z-index:9999;
            background:rgba(0,0,0,.85); overflow:auto;
            padding:24px; box-sizing:border-box;
          `;

          const content = document.createElement("div");
          content.id = "historyViewContent";
          content.style.cssText = `
            max-width:850px; margin:30px auto; padding:24px;
            border-radius:14px; background:#142019;
            color:#e8f5e9; overflow-wrap:anywhere;
          `;

          const closeButton = document.createElement("button");
          closeButton.type = "button";
          closeButton.textContent = "Close";
          closeButton.addEventListener("click", () => modal.remove());

          content.appendChild(closeButton);
          modal.appendChild(content);

          modal.addEventListener("click", (event) => {
            if (event.target === modal) modal.remove();
          });

          document.body.appendChild(modal);
        }

        const content = modal.querySelector("#historyViewContent");

        // Keep the Close button, then rebuild the saved analysis view.
        content.replaceChildren();

        const closeButton = document.createElement("button");
        closeButton.type = "button";
        closeButton.textContent = "Close";
        closeButton.addEventListener("click", () => modal.remove());
        content.appendChild(closeButton);

        const heading = document.createElement("h2");
        heading.textContent = entry.name || "Saved Analysis";
        content.appendChild(heading);

        const savedDate = document.createElement("p");
        savedDate.textContent = "Saved: " + (entry.savedAt || "Unknown");
        content.appendChild(savedDate);

        const summary = document.createElement("p");
        summary.textContent =
          `BPM: ${entry.bpm} | Duration: ${entry.duration}s | ` +
          `Beats: ${entry.beatCount} | Energy: ${entry.rms}`;
        content.appendChild(summary);

        if (entry.snapshot) {
          const snapshot = document.createElement("div");
          snapshot.innerHTML = entry.snapshot;
          content.appendChild(snapshot);
        } else {
          const note = document.createElement("p");
          note.textContent =
            "This saved entry contains summary details only. " +
            "Run the analysis again to view the complete results.";
          content.appendChild(note);
        }

        if (entry.audioUrl) {
          const audio = document.createElement("audio");
          audio.controls = true;
          audio.src = entry.audioUrl;
          audio.style.cssText = "width:100%;margin-top:16px";
          content.appendChild(audio);
        }
      });

      const removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.textContent = "Delete";

      removeButton.addEventListener("click", () => {
        const next = readHistory();
        next.splice(index, 1);

        try {
          localStorage.setItem(historyKey, JSON.stringify(next));
          renderHistory();
        } catch (e) {
          if (saveStatus) {
            saveStatus.textContent = "Could not update saved history.";
          }
        }
      });

      wrap.append(title, details, viewButton, removeButton);
      historyList.appendChild(wrap);
    });
  }

  if (saveButton) {
    saveButton.addEventListener("click", () => {
      let data = {};

      try {
        data = JSON.parse(
          document.getElementById("music-analysis-data")?.textContent || "{}"
        );
      } catch (e) {
        if (saveStatus) {
          saveStatus.textContent = "Could not read the analysis data.";
        }
        return;
      }

      const fileElement = document.querySelector(
        "#analysis + .panel p strong"
      );

      const entry = {
        name:
          fileElement?.parentElement?.textContent
            ?.replace("File:", "")
            .trim() || "Uploaded track",
        savedAt: new Date().toLocaleString(),
        bpm: data.bpm ?? "n/a",
        duration: data.duration ?? "n/a",
        beatCount: data.beat_count ?? "n/a",
        rms: data.rms_energy ?? "n/a",
        audioUrl: player?.currentSrc || "",
        snapshot: resultsPanel?.innerHTML || ""
      };

      const items = readHistory();
      items.unshift(entry);

      try {
        localStorage.setItem(
          historyKey,
          JSON.stringify(items.slice(0, 20))
        );

        renderHistory();

        if (saveStatus) {
          saveStatus.textContent = "Analysis saved in this browser.";
        }
      } catch (e) {
        if (saveStatus) {
          saveStatus.textContent =
            "Could not save history. Browser storage may be full.";
        }
      }
    });
  }

  if (clearButton) {
    clearButton.addEventListener("click", () => {
      if (!confirm("Clear all saved analyses?")) return;

      localStorage.removeItem(historyKey);
      renderHistory();

      if (saveStatus) {
        saveStatus.textContent = "Saved history cleared.";
      }
    });
  }

  renderHistory();
})();


const player=document.getElementById("player");
const canvas=document.getElementById("visualizer");
const button=document.getElementById("visualButton");
const micButton=document.getElementById("micButton");
const statusText=document.getElementById("visualStatus");

if(canvas && button){
 const ctx=canvas.getContext("2d");
 let data={};

 try {
   data=JSON.parse(document.getElementById("music-analysis-data")?.textContent||"{}");
 } catch(e) {
   console.error(e);
 }

 const beats=Array.isArray(data.beat_times)?data.beat_times:[];
 const refEvents=Array.isArray(data.intensity_events)?data.intensity_events:[];
 const curve=Array.isArray(data.rms_curve)?data.rms_curve:[];

 let audioContext=null;
 let analyser=null;
 let trackSource=null;
 let micSource=null;
 let micStream=null;
 let frequencyData=null;
 let enabled=false;
 let micMode=false;
 let pulse=0;
 let lastBeat=-1;
 let energyHistory=[];
 let intensityMode="normal";
 let modeHold=0;
 let intensityEvents=[];
 let lastEventTime=-10;
 let liveTotal=0;
 let liveMatches=0;

 const eventLog=document.getElementById("eventLog");
 const eventStatus=document.getElementById("eventStatus");
 const timeline=document.getElementById("timeline");
 const refList=document.getElementById("refEvents");
 const agreementText=document.getElementById("agreement");

 function formatTime(seconds){
   const minutes=Math.floor(seconds/60);
   const secs=(seconds%60).toFixed(1).padStart(4,"0");
   return `${String(minutes).padStart(2,"0")}:${secs}`;
 }

 function totalDuration(){
   return player && player.duration && isFinite(player.duration)
     ?player.duration:(data.duration||1);
 }

 function seekTo(time){
   if(!player)return;
   player.currentTime=time;
   player.play().catch(()=>{});
 }

 function updateAgreement(){
   const onset=typeof data.beat_onset_agreement==="number"
     ?`${(data.beat_onset_agreement*100).toFixed(0)}% of detected beats fall within 70 ms of an onset peak. This is a consistency check, not independent proof.`
     :"";

   const live=liveTotal>0
     ?`Live detector vs reference: ${liveMatches} of ${liveTotal} live events fall inside a reference region of the same type (±1 s). `
     :"Live detector vs reference: play the song with track visuals enabled to compute agreement. ";

   if(agreementText)agreementText.textContent=live+onset;
 }

 function buildRefList(){
   if(!refList||!refEvents.length)return;
   refList.replaceChildren();

   for(const ev of refEvents){
     const item=document.createElement("button");
     item.type="button";
     item.className="event-item";
     const kind=ev.type==="peak"?"High-intensity region":"Low-intensity region";
     const beatInfo=ev.nearest_beat!==null&&ev.nearest_beat!==undefined
       ?` | Nearest beat: ${formatTime(ev.nearest_beat)}`:"";
     item.textContent=`${formatTime(ev.start)} — ${kind} (${(ev.end-ev.start).toFixed(1)} s)${beatInfo}`;
     item.addEventListener("click",()=>seekTo(ev.start));
     refList.appendChild(item);
   }
 }

 function drawTimeline(){
   if(!timeline||!player)return;
   const w=timeline.clientWidth;
   const h=timeline.clientHeight;
   if(!w||!h)return;
   const tctx=timeline.getContext("2d");
   const total=totalDuration();

   tctx.fillStyle="#0b1917";
   tctx.fillRect(0,0,w,h);

   for(const ev of refEvents){
     tctx.fillStyle=ev.type==="peak"?"rgba(255,155,84,.30)":"rgba(121,170,255,.30)";
     const x=ev.start/total*w;
     tctx.fillRect(x,0,Math.max(2,(ev.end-ev.start)/total*w),h);
   }

   tctx.strokeStyle="rgba(182,216,192,.18)";
   tctx.lineWidth=1;

   for(const b of beats){
     const x=b/total*w;
     tctx.beginPath();
     tctx.moveTo(x,h-10);
     tctx.lineTo(x,h);
     tctx.stroke();
   }

   if(curve.length>1){
     tctx.beginPath();
     curve.forEach((v,i)=>{
       const x=i/(curve.length-1)*w;
       const y=h-8-v*(h-24);
       if(i===0)tctx.moveTo(x,y);
       else tctx.lineTo(x,y);
     });
     tctx.strokeStyle="#b6d8c0";
     tctx.lineWidth=1.5;
     tctx.stroke();
   }

   const px=Math.min(1,player.currentTime/total)*w;
   tctx.strokeStyle="#f2f5ec";
   tctx.lineWidth=2;
   tctx.beginPath();
   tctx.moveTo(px,0);
   tctx.lineTo(px,h);
   tctx.stroke();
 }

 function recordIntensityEvent(mode,time){
   if(!eventLog||time-lastEventTime<0.8)return;
   lastEventTime=time;

   let nearestBeat=null;
   for(const beat of beats){
     if(nearestBeat===null||Math.abs(beat-time)<Math.abs(nearestBeat-time)){
       nearestBeat=beat;
     }
   }

   const label=mode==="peak"?"High-intensity peak":"Low-intensity region";
   liveTotal++;

   if(refEvents.some(e=>e.type===mode&&time>=e.start-1&&time<=e.end+1)){
     liveMatches++;
   }
   updateAgreement();

   intensityEvents.unshift({time,label,nearestBeat});
   intensityEvents=intensityEvents.slice(0,12);
   eventLog.replaceChildren();

   for(const event of intensityEvents){
     const item=document.createElement("button");
     item.type="button";
     item.className="event-item";
     const beatInfo=event.nearestBeat!==null
       ?` | Nearest beat: ${formatTime(event.nearestBeat)}`:"";
     item.textContent=`${formatTime(event.time)} — ${event.label}${beatInfo}`;
     item.addEventListener("click",()=>seekTo(event.time));
     eventLog.appendChild(item);
   }

   if(eventStatus){
     eventStatus.textContent=`${label} detected at ${formatTime(time)}. Click an event to replay it.`;
   }
 }

 function resize(){
   const rect=canvas.getBoundingClientRect();
   const dpr=window.devicePixelRatio||1;
   canvas.width=Math.max(1,Math.floor(rect.width*dpr));
   canvas.height=Math.max(1,Math.floor(rect.height*dpr));
   ctx.setTransform(dpr,0,0,dpr,0,0);
 }

 function resizeTimeline(){
   if(!timeline)return;
   const rect=timeline.getBoundingClientRect();
   const dpr=window.devicePixelRatio||1;
   timeline.width=Math.max(1,Math.floor(rect.width*dpr));
   timeline.height=Math.max(1,Math.floor(rect.height*dpr));
   timeline.getContext("2d").setTransform(dpr,0,0,dpr,0,0);
   drawTimeline();
 }

 resize();
 resizeTimeline();
 buildRefList();
 updateAgreement();
 window.addEventListener("resize",()=>{resize();resizeTimeline();});

 if(timeline){
   timeline.addEventListener("click",ev=>{
     if(!player)return;
     const rect=timeline.getBoundingClientRect();
     const total=totalDuration();
     player.currentTime=Math.max(0,Math.min(total,(ev.clientX-rect.left)/rect.width*total));
     drawTimeline();
   });
 }

 if(player){
   player.addEventListener("timeupdate",drawTimeline);
   player.addEventListener("loadedmetadata",drawTimeline);

   player.addEventListener("seeked",()=>{
     energyHistory=[];
     intensityMode="normal";
     modeHold=0;
     pulse=0;

     let beatIndex=-1;
     while(beatIndex+1<beats.length&&beats[beatIndex+1]<=player.currentTime){
       beatIndex++;
     }
     lastBeat=beatIndex;
     lastEventTime=player.currentTime-1;
     drawTimeline();
   });
 }

 async function ensureAudioContext(){
   if(!audioContext){
     audioContext=new(window.AudioContext||window.webkitAudioContext)();
     analyser=audioContext.createAnalyser();
     analyser.fftSize=256;
     analyser.smoothingTimeConstant=.72;
     frequencyData=new Uint8Array(analyser.frequencyBinCount);
   }
   await audioContext.resume();
 }

 button.addEventListener("click",async()=>{
   try{
     if(!player){
       if(statusText)statusText.textContent="Upload a song before enabling track visuals.";
       return;
     }

     await ensureAudioContext();

     if(micMode){
       micMode=false;
       if(micSource){try{micSource.disconnect();}catch(e){}micSource=null;}
       if(micStream){micStream.getTracks().forEach(t=>t.stop());micStream=null;}
       if(trackSource){try{trackSource.disconnect();}catch(e){}trackSource=null;}
     }

     
if(!trackSource){
    trackSource=audioContext.createMediaElementSource(player);
    trackSource.connect(analyser);
    analyser.connect(audioContext.destination);
}

if(audioContext.state === "suspended"){
    await audioContext.resume();
}

enabled=true;
button.disabled=true;
button.textContent="Track Visuals Enabled";
if(micButton)micButton.textContent="Start Microphone Visuals";
if(statusText)statusText.textContent="Track visuals enabled. Play your song!";
draw();

   }catch(e){
     console.error(e);
     if(statusText)statusText.textContent="Could not enable visuals. Refresh the page and try again.";
   }
 });

 
if (micButton) {
  micButton.addEventListener("click", async () => {
    try {
      await ensureAudioContext();

      if (!micMode) {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("Microphone API unavailable");
        }

        micStream = await navigator.mediaDevices.getUserMedia({
          audio: true
        });

        micSource = audioContext.createMediaStreamSource(micStream);
        micSource.connect(analyser);

        micMode = true;
        enabled = true;
        energyHistory = [];
        intensityMode = "normal";
        modeHold = 0;

        micButton.textContent = "Stop Microphone Visuals";
        button.disabled = false;
        button.textContent = "Switch to Track Visuals";

        if (statusText) {
          statusText.textContent =
            "Microphone visuals active. Speak, sing, or play audio near your microphone.";
        }

        draw();

      } else {
        micMode = false;

        if (micSource) {
          try {
            micSource.disconnect();
          } catch (e) {}
          micSource = null;
        }

        if (micStream) {
          micStream.getTracks().forEach(track => track.stop());
          micStream = null;
        }

        micButton.textContent = "Start Microphone Visuals";

        if (audioContext.state === "suspended") {
          await audioContext.resume();
        }

        button.textContent = "Switch to Track Visuals";
        button.disabled = false;

        if (statusText) {
          statusText.textContent =
            "Microphone stopped. Press play to continue your song.";
        }
      }

    } catch (e) {
      console.error(e);

      if (statusText) {
        statusText.textContent =
          "Microphone unavailable. Allow microphone permission and try again.";
      }
    }
  });
}


 function draw(){
   if(!enabled||!analyser||!frequencyData)return;
   requestAnimationFrame(draw);

   const w=canvas.clientWidth;
   const h=canvas.clientHeight;
   if(!w||!h)return;

   analyser.getByteFrequencyData(frequencyData);
   const energy=frequencyData.reduce((a,b)=>a+b,0)/frequencyData.length;

   let avg=energy;
   let variance=0;

   if(energyHistory.length>10){
     avg=energyHistory.reduce((a,b)=>a+b,0)/energyHistory.length;
     variance=energyHistory.reduce((sum,value)=>sum+(value-avg)**2,0)/energyHistory.length;
   }

   const deviation=Math.sqrt(variance);
   let detectedMode="normal";

   if(energyHistory.length>10){
     if(energy>avg+deviation*.5)detectedMode="peak";
     else if(energy<avg-deviation*.5)detectedMode="low";
   }

   if(detectedMode===intensityMode){
     modeHold=0;
   }else{
     modeHold++;
     if(modeHold>=3){
       const previousMode=intensityMode;
       intensityMode=detectedMode;
       modeHold=0;

       if(!micMode&&player&&!player.paused&&intensityMode!=="normal"&&intensityMode!==previousMode){
         recordIntensityEvent(intensityMode,player.currentTime);
       }
     }
   }

   energyHistory.push(energy);
   if(energyHistory.length>120)energyHistory.shift();

   if(!micMode&&player&&!player.paused&&!player.ended){
     let i=0;
     while(i<beats.length&&beats[i]<=player.currentTime)i++;
     const current=i-1;
     if(current>=0&&current!==lastBeat){
       if(lastBeat<0||current>lastBeat)pulse=1;
       lastBeat=current;
     }
   }

   pulse*=.91;
   const hue=intensityMode==="peak"?25:intensityMode==="low"?220:135;
   const modeLabel=intensityMode==="peak"?"HIGH INTENSITY PEAK":
     intensityMode==="low"?"LOW INTENSITY":"STEADY INTENSITY";
   const modeColor=intensityMode==="peak"?"#ff9b54":
     intensityMode==="low"?"#79aaff":"#b6d8c0";

   ctx.fillStyle="#0b1917";
   ctx.fillRect(0,0,w,h);
   ctx.fillStyle=modeColor;
   ctx.font="bold 12px system-ui";
   ctx.fillText(micMode?"MICROPHONE VISUALS · "+modeLabel:modeLabel,14,22);

   const radius=35+energy*.65+pulse*35;
   const cx=w/2;
   const cy=h/2;

   for(let ring=4;ring>=0;ring--){
     ctx.beginPath();
     ctx.arc(cx,cy,radius+ring*20,0,Math.PI*2);
     ctx.strokeStyle=`hsla(${(hue+ring*18)%360},45%,${65-ring*3}%,${.8-ring*.1})`;
     ctx.lineWidth=2+pulse*2;
     ctx.stroke();
   }

   const count=48;
   const bw=w/count;

   for(let i=0;i<count;i++){
     const value=frequencyData[i]||0;
     const bh=value/255*h*.38;
     const barIntensity=value/255;
     const barHue=intensityMode==="peak"?25-barIntensity*12:
       intensityMode==="low"?220-barIntensity*15:
       135-barIntensity*25;

     ctx.fillStyle=`hsl(${barHue},55%,${35+barIntensity*30}%)`;
     ctx.fillRect(i*bw+1,cy-bh/2,Math.max(1,bw-3),bh);
   }

   ctx.beginPath();
   ctx.arc(cx,cy,8+energy*.12+pulse*15,0,Math.PI*2);
   ctx.fillStyle="#c3e7ce";
   ctx.fill();
 }
}

document.querySelectorAll(".nav-link").forEach(link=>{
 link.addEventListener("click",()=>{
   document.querySelectorAll(".nav-link").forEach(x=>x.classList.remove("active"));
   link.classList.add("active");
 });
});

const compareModeSelect=document.getElementById("compareMode");
const intentWrap=document.getElementById("intentWrap");

function updateIntentVisibility(){
 if(compareModeSelect&&intentWrap){
   intentWrap.hidden=compareModeSelect.value!=="revision";
 }
}
if(compareModeSelect){
 compareModeSelect.addEventListener("change",updateIntentVisibility);
 updateIntentVisibility();
}

if(location.pathname==="/compare"){
 const target=document.getElementById("compare");
 if(target)target.scrollIntoView();
}
</script>
</body>
</html>
"""
INTENTS = {
    "none": ("No specific intent (just show the changes)", None),
    "more_energy": ("I intended to increase signal energy (RMS +5% or more)",
                    lambda a, b: b["rms"] >= a["rms"] * 1.05),
    "less_energy": ("I intended to reduce signal energy (RMS -5% or more)",
                    lambda a, b: b["rms"] <= a["rms"] * 0.95),
    "brighter": ("I intended a brighter sound (centroid +5% or more)",
                 lambda a, b: b["centroid"] >= a["centroid"] * 1.05),
    "darker": ("I intended a darker sound (centroid -5% or more)",
               lambda a, b: b["centroid"] <= a["centroid"] * 0.95),
    "same_tempo": ("I intended to keep the tempo (within 2 BPM)",
                   lambda a, b: abs(b["bpm"] - a["bpm"]) <= 2),
}


def page(**kwargs):
    context = dict(
        message=None, audio_url=None, audio_type=None, results=None,
        song_name=None, comparison=None, compare_message=None,
        compare_mode="revision", selected_intent="none",
        intents=[(k, v[0]) for k, v in INTENTS.items()],
    )
    context.update(kwargs)
    return render_template_string(HTML, **context)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"mp3", "wav"}


def file_too_big(storage):
    storage.stream.seek(0, 2)
    size = storage.stream.tell()
    storage.stream.seek(0)
    return size > MAX_FILE_BYTES


def generate_suggestions(bpm, beat_count, duration):
    suggestions = []

    if bpm < 70:
        suggestions.append(
            f"The estimated tempo is {bpm} BPM, which is relatively slow."
        )
    elif bpm < 100:
        suggestions.append(
            f"The estimated tempo is {bpm} BPM. Consider percussion or syncopated rhythms."
        )
    elif bpm < 130:
        suggestions.append(
            f"The estimated tempo is {bpm} BPM, a moderate-to-fast tempo."
        )
    else:
        suggestions.append(
            f"The estimated tempo is {bpm} BPM, which is relatively fast."
        )

    if duration > 0:
        density = beat_count / duration
        suggestions.append(
            f"Approximately {density:.2f} beats were detected per second. "
            "Compare the timestamps with playback to assess accuracy."
        )

    if duration < 60:
        suggestions.append("This is a short track or audio excerpt.")
    elif duration >= 180:
        suggestions.append("Compare different sections to explore rhythmic changes.")

    suggestions.append(
        "Tempo and beat suggestions are rule-based. Emotion scores are estimates and require validation."
    )
    return suggestions


def generate_development_pathway(bpm, beat_count, duration, rms, centroid, zcr):
    density = beat_count / duration if duration > 0 else 0

    if bpm < 70:
        rhythm_advice = "Explore whether a stronger rhythmic pulse or additional percussion would suit your musical goal."
    elif bpm < 100:
        rhythm_advice = "Experiment with syncopation or percussion patterns while preserving the song's groove."
    elif bpm < 130:
        rhythm_advice = "Experiment with rhythmic variation and arrangement changes around the detected tempo."
    else:
        rhythm_advice = "Check whether the fast tempo supports the intended feel, and experiment with rhythmic spacing."

    if rms < 0.05:
        energy_advice = "The average signal energy is low. Experiment with layering or volume balance if greater intensity is desired."
    elif rms < 0.15:
        energy_advice = "The average signal energy is moderate. Compare different sections to explore changes in intensity."
    else:
        energy_advice = "The average signal energy is high. Check that louder elements remain balanced and clear."

    if centroid < 1500:
        brightness_advice = "The average spectral centroid is low. Experiment with brighter elements if they suit the intended sound."
    elif centroid < 3000:
        brightness_advice = "The average spectral centroid indicates moderate brightness. Experiment with tonal contrast between elements."
    else:
        brightness_advice = "The average spectral centroid is high. Check whether the brighter sound remains clear and comfortable."

    if zcr > 0.15:
        texture_advice = "The zero-crossing rate is relatively high. Listen for noisy or sharp textures and decide whether they fit the arrangement."
    else:
        texture_advice = "Use the zero-crossing rate as an additional texture indicator, and confirm your interpretation by listening."

    return [
        {
            "title": "Review the rhythm",
            "detail": f"Estimated tempo: {bpm} BPM, with about {density:.2f} detected beats per second. {rhythm_advice}"
        },
        {
            "title": "Review overall intensity",
            "detail": f"Average RMS energy: {rms:.4f}. {energy_advice}"
        },
        {
            "title": "Explore spectral brightness",
            "detail": f"Spectral centroid: {centroid:.2f} Hz. {brightness_advice}"
        },
        {
            "title": "Review the sound texture",
            "detail": f"Zero-crossing rate: {zcr:.4f}. {texture_advice}"
        },
        {
            "title": "Compare your revisions",
            "detail": "Change one musical element at a time, listen to the result, and compare the tempo and acoustic measurements before and after (use Compare Two Versions below). These measurements guide exploration; they do not automatically determine musical quality."
        }
    ]


def mood_label(valence, arousal):
    if valence >= 6:
        valence_label = "Higher Valence Estimate"
    elif valence <= 4:
        valence_label = "Lower Valence Estimate"
    else:
        valence_label = "Mid-range Valence Estimate"

    if arousal >= 5.5:
        energy_label = "Higher Arousal / Energy"
    elif arousal <= 4:
        energy_label = "Lower Arousal / Energy"
    else:
        energy_label = "Moderate Arousal / Energy"

    return f"{valence_label} · {energy_label}"


@lru_cache(maxsize=1)
def load_emotion_assets():
    """Load model, metadata, openSMILE extractor and DEAM reference ranges once."""
    if opensmile is None:
        raise RuntimeError("openSMILE is not installed in this Python environment.")

    if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
        raise RuntimeError("Model or dataset/columns.csv was not found.")

    bundle = joblib.load(MODEL_PATH)
    feature_columns = bundle["feature_columns"]

    metadata = pd.read_csv(COLUMNS_PATH).fillna("")
    metadata = metadata[metadata["column"].isin(feature_columns)].copy()

    reference = pd.read_csv(
        REFERENCE_PATH, usecols=feature_columns
    ).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    return {
        "bundle": bundle,
        "model": bundle["model"],
        "imputer": bundle["imputer"],
        "feature_columns": feature_columns,
        "target_columns": bundle.get("target_columns", ["valence_mean", "arousal_mean"]),
        "metadata_by_name": metadata.set_index("column").to_dict("index"),
        "lower": reference.quantile(0.01),
        "upper": reference.quantile(0.99),
        "smile": smile,
    }


def extract_emotion_features(filepath):
    """Approximate the model's feature vector. Pipeline equivalence is not validated."""
    assets = load_emotion_assets()
    feature_columns = assets["feature_columns"]
    metadata_by_name = assets["metadata_by_name"]

    frames = assets["smile"].process_file(str(filepath)).reset_index(drop=True)

    if frames.empty:
        raise RuntimeError("No audio frames were extracted.")

    def find_source_column(descriptor):
        descriptor = str(descriptor).strip()
        candidates = [descriptor, descriptor + "_sma"]

        if descriptor.startswith("pcm_fftMag_mfcc["):
            index = descriptor.split("[", 1)[1].split("]", 1)[0]
            candidates.extend([f"mfcc_sma[{index}]", f"mfcc[{index}]"])

        if "[" in descriptor:
            base, rest = descriptor.split("[", 1)
            candidates.extend([base + "_sma[" + rest, base + "[" + rest])

        for name in candidates:
            if name in frames.columns:
                return name
        return None

    raw = {}
    missing_descriptors = []

    descriptors = sorted({
        str(row.get("low_level_descriptor", "")).strip()
        for row in metadata_by_name.values()
        if str(row.get("low_level_descriptor", "")).strip()
    })

    for descriptor in descriptors:
        source_column = find_source_column(descriptor)
        if source_column is None:
            missing_descriptors.append(descriptor)
            continue

        values = pd.to_numeric(frames[source_column], errors="coerce").to_numpy(float)
        raw[(descriptor, "original")] = values
        raw[(descriptor, "delta")] = np.diff(values, prepend=values[0])

    if missing_descriptors:
        raise RuntimeError(
            "openSMILE descriptors do not match the dataset: "
            + ", ".join(missing_descriptors[:5])
        )

    frames_per_window = 50
    window_ids = np.arange(len(frames)) // frames_per_window
    window_stats = []

    for window_id in np.unique(window_ids):
        mask = window_ids == window_id
        stats = {}

        for (descriptor, kind), values in raw.items():
            chunk = values[mask]
            chunk = chunk[np.isfinite(chunk)]
            stats[(descriptor, kind, "mean")] = float(np.mean(chunk)) if len(chunk) else np.nan
            stats[(descriptor, kind, "std")] = float(np.std(chunk)) if len(chunk) else np.nan

        window_stats.append(stats)

    def norm_kind(value):
        return "delta" if str(value).strip().lower() in {
            "delta", "de", "derivative", "difference"
        } else "original"

    def norm_stat(value):
        return "std" if str(value).strip().lower() in {
            "std", "stddev", "standard deviation", "standard_deviation"
        } else "mean"

    vector = []
    missing = []

    for feature in feature_columns:
        row = metadata_by_name.get(feature)

        if not row:
            missing.append(feature)
            vector.append(np.nan)
            continue

        descriptor = str(row.get("low_level_descriptor", "")).strip()
        kind = norm_kind(row.get("value_or_delta", ""))
        stat = norm_stat(row.get("window_statistic", ""))
        aggregation = norm_stat(row.get("song_aggregation", ""))

        values = np.asarray([
            window.get((descriptor, kind, stat), np.nan)
            for window in window_stats
        ], dtype=float)
        values = values[np.isfinite(values)]

        if not len(values):
            missing.append(feature)
            vector.append(np.nan)
        elif aggregation == "std":
            vector.append(float(np.std(values)))
        else:
            vector.append(float(np.mean(values)))

    if missing:
        raise RuntimeError(
            f"Could not calculate {len(missing)} expected model features."
        )

    X = pd.DataFrame([vector], columns=feature_columns)
    return X.replace([np.inf, -np.inf], np.nan)


def predict_emotion(filepath):
    assets = load_emotion_assets()
    X = extract_emotion_features(filepath)

    current = X.iloc[0]
    lower, upper = assets["lower"], assets["upper"]
    comparable = current.notna() & lower.notna() & upper.notna()
    outside = comparable & ((current < lower) | (current > upper))
    checked = int(comparable.sum())
    outside_count = int(outside.sum())
    outside_pct = 100 * outside_count / checked if checked else 100.0
    imputed = int(current.isna().sum())

    if outside_pct < 10:
        reliability = "Most features fall inside the DEAM range"
    elif outside_pct < 30:
        reliability = "Some features fall outside the DEAM range - interpret with caution"
    else:
        reliability = "Many features fall outside the DEAM range - low confidence"

    app.logger.warning(
        "SONIQ feature-range check: %d/%d features (%.1f%%) "
        "outside the DEAM 1st-99th percentile range; %d imputed.",
        outside_count, checked, outside_pct, imputed
    )

    prediction = assets["model"].predict(assets["imputer"].transform(X))[0]
    values = dict(zip(assets["target_columns"], prediction))
    valence = float(np.clip(values["valence_mean"], 1, 9))
    arousal = float(np.clip(values["arousal_mean"], 1, 9))

    return {
        "valence": round(valence, 2),
        "arousal": round(arousal, 2),
        "mood": mood_label(valence, arousal),
        "range_checked": checked,
        "range_outside": outside_count,
        "range_pct": round(outside_pct, 1),
        "imputed": imputed,
        "reliability": reliability,
    }


def measure_audio(filepath):
    y, sr = librosa.load(str(filepath), sr=22050, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio file.")

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    measures = {
        "bpm": round(float(np.asarray(tempo).reshape(-1)[0]), 1),
        "duration": round(len(y) / sr, 2),
        "beat_count": len(beat_times),
        "rms": float(np.mean(librosa.feature.rms(y=y)[0])),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(y)[0])),
    }
    return y, sr, beat_times, measures


def compute_intensity_events(y, sr, beat_times):
    """Find relative high/low-intensity regions from the RMS envelope."""
    hop = 512
    fps = sr / hop
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]

    if len(rms) == 0:
        return [], []

    db = librosa.amplitude_to_db(rms, ref=max(float(rms.max()), 1e-9))
    smooth = min(len(db), max(1, int(0.5 * fps)))
    env = np.convolve(db, np.ones(smooth) / smooth, mode="same")

    rolling = pd.Series(env).rolling(
        int(4 * fps), center=True, min_periods=max(1, int(fps))
    )
    mean = rolling.mean().to_numpy()
    std = np.maximum(rolling.std().fillna(0).to_numpy(), 1.0)

    state = np.zeros(len(env), dtype=int)
    state[env > mean + 0.75 * std] = 1
    state[env < mean - 0.75 * std] = -1

    events = []
    i = 0

    while i < len(state):
        j = i
        while j < len(state) and state[j] == state[i]:
            j += 1

        if state[i] != 0 and (j - i) / fps >= 0.5:
            start, end = i / fps, j / fps
            nearest = None

            if len(beat_times):
                nearest = round(float(beat_times[np.argmin(np.abs(beat_times - start))]), 2)

            events.append({
                "type": "peak" if state[i] > 0 else "low",
                "start": round(start, 2),
                "end": round(end, 2),
                "nearest_beat": nearest,
            })
        i = j

    span = float(env.max() - env.min()) or 1.0
    norm = (env - env.min()) / span
    idx = np.linspace(0, len(norm) - 1, min(600, len(norm))).astype(int)
    curve = [round(float(v), 3) for v in norm[idx]]

    return events[:300], curve


def beat_onset_agreement(y, sr, beat_times):
    """Consistency check: beats within 70 ms of an onset peak."""
    if not len(beat_times):
        return None

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units="time"
    )

    if not len(onsets):
        return None

    distance = np.abs(beat_times[:, None] - onsets[None, :]).min(axis=1)
    return round(float(np.mean(distance <= 0.07)), 3)


def analyse_audio(filepath):
    y, sr, beat_times, m = measure_audio(filepath)
    bpm, duration, beat_count = m["bpm"], m["duration"], m["beat_count"]
    rms, centroid, zcr = m["rms"], m["centroid"], m["zcr"]

    energy = (
        "Low average signal energy" if rms < 0.05 else
        "Moderate average signal energy" if rms < 0.15 else
        "High average signal energy"
    )

    brightness = (
        "Darker spectral character" if centroid < 1500 else
        "Moderate spectral brightness" if centroid < 3000 else
        "Brighter spectral character"
    )

    events, curve = compute_intensity_events(y, sr, beat_times)

    results = {
        "bpm": bpm,
        "duration": duration,
        "beat_count": beat_count,
        "beat_times": [round(float(t), 3) for t in beat_times],
        "timestamps": ", ".join(f"{t:.2f}" for t in beat_times[:200]) or "No beats detected",
        "rms_energy": round(rms, 4),
        "spectral_centroid_hz": round(centroid, 2),
        "zero_crossing_rate": round(zcr, 4),
        "energy_description": energy,
        "brightness_description": brightness,
        "intensity_events": events,
        "rms_curve": curve,
        "beat_onset_agreement": beat_onset_agreement(y, sr, beat_times),
        "suggestions": generate_suggestions(bpm, beat_count, duration),
        "development_pathway": generate_development_pathway(
            bpm, beat_count, duration, rms, centroid, zcr
        ),
        "emotion": None,
        "emotion_error": None,
    }

    try:
        results["emotion"] = predict_emotion(filepath)
    except Exception as exc:
        app.logger.exception("Emotion prediction unavailable")
        results["emotion_error"] = str(exc)

    return results
def analyse_for_compare(filepath):
    _, _, _, m = measure_audio(filepath)

    try:
        emotion = predict_emotion(filepath)
        m["valence"], m["arousal"] = emotion["valence"], emotion["arousal"]
    except Exception:
        app.logger.exception("Emotion prediction unavailable (compare)")
        m["valence"] = m["arousal"] = None

    return m


def build_comparison(
    a, b, intent, compare_mode="revision",
    a_name="Version A", b_name="Version B"
):
    def change(x, y, digits):
        if x is None or y is None:
            return "n/a"

        diff = y - x
        pct = f" ({diff / x * 100:+.1f}%)" if x else ""
        return f"{diff:+.{digits}f}{pct}"

    spec = [
        ("Tempo (BPM)", "bpm", 1),
        ("Duration (s)", "duration", 2),
        ("Detected beats", "beat_count", 0),
        ("Average RMS energy", "rms", 4),
        ("Spectral centroid (Hz)", "centroid", 1),
        ("Zero-crossing rate", "zcr", 4),
        ("Predicted valence (/9)", "valence", 2),
        ("Predicted arousal (/9)", "arousal", 2),
    ]

    rows = []

    for label, key, digits in spec:
        va, vb = a.get(key), b.get(key)
        fmt = lambda v: "n/a" if v is None else f"{v:.{digits}f}"

        rows.append({
            "label": label,
            "a": fmt(va),
            "b": fmt(vb),
            "change": change(va, vb, digits),
        })

    intent = intent if intent in INTENTS else "none"

    if compare_mode == "tracks":
        summary_label = "Comparison note"
        check = (
            "Descriptive comparison only. These measurements show differences "
            "between the uploaded tracks; they do not rank quality or establish "
            "which track is better."
        )
        note = (
            "Choose this mode for different songs. BPM, loudness and spectral "
            "features depend on the recording and mastering. Emotion values are "
            "estimates from an unvalidated inference feature pipeline."
        )
    else:
        summary_label = "Revision check"
        test = INTENTS[intent][1]

        if test is None:
            check = "No revision goal selected - review the measured changes above."
        else:
            check = ("Met: " if test(a, b) else "Not met: ") + INTENTS[intent][0]

        note = (
            "Use this mode for two revisions of the same song. SONIQ does not "
            "verify that the uploads are the same song. Tempo estimates can jump "
            "by half or double, so confirm BPM changes by listening. Emotion "
            "changes are estimates from an unvalidated inference feature pipeline. "
            "A measured target being met does not prove the revision sounds better."
        )

    def short_name(name):
        name = str(name or "Audio")
        return name if len(name) <= 36 else name[:33] + "..."

    return {
        "rows": rows,
        "check": check,
        "note": note,
        "mode": compare_mode,
        "summary_label": summary_label,
        "a_name": short_name(a_name),
        "b_name": short_name(b_name),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    message = None
    audio_url = None
    audio_type = None
    results = None
    filename = None

    if request.method == "POST":
        uploaded = request.files.get("audio")

        if not uploaded or not uploaded.filename:
            message = "Please select a song."
        elif not allowed_file(uploaded.filename):
            message = "Only MP3 and WAV files are supported."
        elif file_too_big(uploaded):
            message = "File too large. Maximum size is 20 MB."
        else:
            filename = secure_filename(uploaded.filename)

            if not filename:
                message = "Invalid filename."
            else:
                stored_filename = f"{uuid.uuid4().hex[:12]}_{filename}"
                filepath = UPLOAD_FOLDER / stored_filename

                try:
                    uploaded.save(str(filepath))
                    results = analyse_audio(filepath)
                    message = "Song uploaded and analysed successfully!"
                    audio_url = f"/uploads/{stored_filename}"
                    audio_type = (
                        "audio/mpeg" if filename.lower().endswith(".mp3")
                        else "audio/wav"
                    )
                except Exception:
                    app.logger.exception("Audio analysis failed")
                    message = (
                        "Could not analyse this file. "
                        "Try another valid MP3 or WAV file."
                    )
                    results = None
                    audio_url = None

    return page(
        message=message,
        audio_url=audio_url,
        audio_type=audio_type,
        results=results,
        song_name=filename if audio_url else None,
    )


@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "GET":
        return redirect("/#compare")

    files = [request.files.get("audio_a"), request.files.get("audio_b")]
    intent = request.form.get("intent", "none")
    compare_mode = request.form.get("compare_mode", "revision")

    if compare_mode not in {"revision", "tracks"}:
        compare_mode = "revision"

    def error(message):
        return page(
            compare_message=message,
            compare_mode=compare_mode,
            selected_intent=intent if intent in INTENTS else "none",
        )

    if any(not f or not f.filename for f in files):
        return error("Please select both audio files.")

    if any(not allowed_file(f.filename) for f in files):
        return error("Only MP3 and WAV files are supported.")

    if any(file_too_big(f) for f in files):
        return error("Each file must be 20 MB or smaller.")

    try:
        measured = []

        for f in files:
            name = secure_filename(f.filename)

            if not name:
                return error("Invalid filename.")

            path = UPLOAD_FOLDER / f"compare_{uuid.uuid4().hex[:12]}_{name}"
            f.save(str(path))
            measured.append((analyse_for_compare(path), name))

        comparison = build_comparison(
            measured[0][0], measured[1][0], intent, compare_mode,
            measured[0][1], measured[1][1],
        )

        return page(
            comparison=comparison,
            compare_mode=compare_mode,
            selected_intent=intent if intent in INTENTS else "none",
        )

    except Exception:
        app.logger.exception("Comparison failed")
        return error("Could not compare these files. Try valid MP3 or WAV files.")


@app.route("/uploads/<path:filename>")
def uploaded_audio(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.errorhandler(413)
def file_too_large(error):
    return "Upload request is too large. Keep each audio file at or below 20 MB.", 413


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)