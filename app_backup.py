
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp3", "wav"}
MAX_FILE_SIZE = 20 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Music Analysis & Beat-Responsive Visual Intelligence</title>
<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    padding: 30px 16px;
    background: #101323;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}

.container {
    max-width: 850px;
    margin: auto;
    padding: 28px;
    background: #202640;
    border-radius: 18px;
}

h1 { color: #aaa5ff; }
p { color: #d5d8ed; }

input, button {
    margin: 10px;
    padding: 12px;
    border-radius: 8px;
}

button {
    background: #8885ff;
    color: white;
    border: 0;
    cursor: pointer;
    font-weight: bold;
}

button:hover { background: #6e69e8; }
button:disabled { opacity: 0.7; cursor: default; }

audio {
    width: 100%;
    max-width: 500px;
    margin: 15px 0;
}

.results, .visual-box {
    background: #303957;
    border-radius: 14px;
    padding: 20px;
    margin-top: 24px;
}

.bpm {
    font-size: 32px;
    color: #a5ffdc;
}

canvas {
    display: block;
    width: 100%;
    height: 300px;
    background: #090d1d;
    border-radius: 12px;
}

.timestamps {
    max-height: 120px;
    overflow-y: auto;
    overflow-wrap: anywhere;
    color: #d5d8ed;
}

.suggestions {
    text-align: left;
    color: #d5d8ed;
}

.suggestions li { margin: 12px 0; }
small { color: #c6cbe2; }

@media (max-width: 600px) {
    .container { padding: 18px 12px; }
    h1 { font-size: 25px; }
}
</style>
</head>

<body>
<div class="container">
    <h1>Music Analysis & Beat-Responsive Visual Intelligence</h1>
    <p>Upload a song, explore its rhythm, and watch the visuals react.</p>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="audio" accept=".mp3,.wav" required>
        <br>
        <button type="submit">Upload & Analyse</button>
    </form>

    {% if message %}
        <p>{{ message }}</p>
    {% endif %}

    {% if audio_url %}
    <div class="results">
        <h2>Your Song</h2>
        <p><strong>File:</strong> {{ song_name }}</p>

        <audio id="player" controls>
            <source src="{{ audio_url }}" type="{{ audio_type }}">
            Your browser does not support audio playback.
        </audio>

        {% if results %}
        <h2>Music Analysis Results</h2>
        <p>Estimated Tempo</p>
        <div class="bpm">{{ results.bpm }} BPM</div>
        <p>Duration: {{ results.duration }} seconds</p>
        <p>Detected Beats: {{ results.beat_count }}</p>

        <h3>Beat Timestamps (seconds)</h3>
        <div class="timestamps">{{ results.timestamps }}</div>

<h3>Audio Feature Analysis</h3>

<div style="text-align: left; color: #d5d8ed;">
    <p>
        <strong>Average RMS Energy:</strong>
        {{ results.rms_energy }}
    </p>

    <p>
        <strong>Spectral Centroid:</strong>
        {{ results.spectral_centroid_hz }} Hz
    </p>

    <p>
        <strong>Zero-Crossing Rate:</strong>
        {{ results.zero_crossing_rate }}
    </p>

    <p>
        <strong>Energy Level:</strong>
        {{ results.energy_description }}
    </p>

    <p>
        <strong>Spectral Brightness:</strong>
        {{ results.brightness_description }}
    </p>
</div>

        <h3>Music Improvement Suggestions</h3>
        <ul class="suggestions">
            {% for suggestion in results.suggestions %}
                <li>{{ suggestion }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>

    <div class="visual-box">
        <h2>Live Beat-Responsive Visualiser</h2>
        <p>Play your song to animate the colours and patterns.</p>

        <canvas id="visualizer"></canvas>

        <p>
            <small>
                Visuals respond to audio frequency, intensity and detected beats.
            </small>
        </p>

        <button id="visualButton" type="button">
            Enable Live Visuals
        </button>

        <p id="visualStatus">Enable visuals, then play your song.</p>
    </div>

    <script id="music-analysis-data" type="application/json">{{ results | tojson if results else '{}' }}</script>
    {% endif %}
</div>

<script>
const player = document.getElementById("player");
const canvas = document.getElementById("visualizer");
const button = document.getElementById("visualButton");
const statusText = document.getElementById("visualStatus");

if (player && canvas && button) {
    const ctx = canvas.getContext("2d");

    const dataElement = document.getElementById("music-analysis-data");
    let analysisData = {};

    try {
        analysisData = JSON.parse(dataElement.textContent || "{}");
    } catch (error) {
        console.error("Could not read analysis data:", error);
    }

    const beatTimes = Array.isArray(analysisData.beat_times)
        ? analysisData.beat_times
        : [];

    let audioContext = null;
    let analyser = null;
    let source = null;
    let frequencyData = null;
    let enabled = false;
    let phase = 0;
    let pulse = 0;
    let lastBeatIndex = -1;
    let animationStarted = false;

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        canvas.height = Math.max(1, Math.floor(rect.height * dpr));

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    player.addEventListener("seeked", () => {
        lastBeatIndex = -1;
        pulse = 0;
    });

    button.addEventListener("click", async () => {
        try {
            if (!audioContext) {
                audioContext = new (
                    window.AudioContext || window.webkitAudioContext
                )();

                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                analyser.smoothingTimeConstant = 0.72;

                frequencyData = new Uint8Array(
                    analyser.frequencyBinCount
                );

                source = audioContext.createMediaElementSource(player);
                source.connect(analyser);
                analyser.connect(audioContext.destination);
            }

            await audioContext.resume();

            enabled = true;
            statusText.textContent =
                "Live visuals enabled. Play your song!";
            button.textContent = "Live Visuals Enabled";
            button.disabled = true;

            if (!animationStarted) {
                animationStarted = true;
                draw();
            }
        } catch (error) {
            console.error("Visualiser error:", error);
            statusText.textContent =
                "Could not enable visuals. Refresh and try again.";
        }
    });

    function draw() {
        if (!enabled) return;

        requestAnimationFrame(draw);

        const w = canvas.clientWidth;
        const h = canvas.clientHeight;

        if (!w || !h) return;

        analyser.getByteFrequencyData(frequencyData);

        let sum = 0;
        for (let i = 0; i < frequencyData.length; i++) {
            sum += frequencyData[i];
        }

        const energy = frequencyData.length
            ? sum / frequencyData.length
            : 0;

        // Trigger pulses using the beat timestamps detected by Librosa.
        if (!player.paused && !player.ended && beatTimes.length > 0) {
            const currentTime = player.currentTime;

            // Binary search for the most recent detected beat.
            let low = 0;
            let high = beatTimes.length;

            while (low < high) {
                const mid = Math.floor((low + high) / 2);

                if (beatTimes[mid] <= currentTime) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            const beatIndex = low - 1;

            if (beatIndex >= 0 && beatIndex !== lastBeatIndex) {
                if (lastBeatIndex < 0 || beatIndex > lastBeatIndex) {
                    pulse = 1;
                }

                lastBeatIndex = beatIndex;
            }
        }

        pulse *= 0.91;
        phase += 0.012 + energy * 0.0003;

        const hue = (phase * 30 + energy * 2) % 360;

        ctx.fillStyle = "#090d1d";
        ctx.fillRect(0, 0, w, h);

        const radius = 35 + energy * 0.65 + pulse * 35;
        const cx = w / 2;
        const cy = h / 2;

        // Concentric rings react to detected beats.
        for (let ring = 4; ring >= 0; ring--) {
            ctx.beginPath();
            ctx.arc(
                cx,
                cy,
                radius + ring * 20,
                0,
                Math.PI * 2
            );

            ctx.strokeStyle =
                `hsla(${(hue + ring * 24) % 360}, 95%, 65%, ${0.8 - ring * 0.1})`;

            ctx.lineWidth = 2 + pulse * 2;
            ctx.stroke();
        }

        // Frequency bars react to the sound spectrum.
        const barCount = 48;
        const barWidth = w / barCount;

        for (let i = 0; i < barCount; i++) {
            const value = frequencyData[i] || 0;
            const barHeight = (value / 255) * h * 0.38;
            const x = i * barWidth;

            ctx.fillStyle =
                `hsl(${(hue + i * 5) % 360}, 90%, 65%)`;

            ctx.fillRect(
                x + 1,
                cy - barHeight / 2,
                Math.max(1, barWidth - 3),
                barHeight
            );
        }

        // Central circle reacts to audio energy and beat pulses.
        ctx.beginPath();
        ctx.arc(
            cx,
            cy,
            8 + energy * 0.12 + pulse * 15,
            0,
            Math.PI * 2
        );

        ctx.fillStyle = `hsl(${hue}, 100%, 75%)`;
        ctx.fill();
    }
}
</script>
</body>
</html>
"""


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )



def generate_suggestions(bpm, beat_count, duration):
    suggestions = []

    if bpm < 70:
        suggestions.append(
            f"The estimated tempo is {bpm} BPM, which is relatively slow. "
            "If you want a more energetic track, experiment with a faster "
            "tempo or add rhythmic percussion."
        )
    elif bpm < 100:
        suggestions.append(
            f"The estimated tempo is {bpm} BPM. Try adding percussion "
            "or syncopated rhythms if you want more movement."
        )
    elif bpm < 130:
        suggestions.append(
            f"The estimated tempo is {bpm} BPM, a moderate-to-fast tempo. "
            "Experiment with rhythmic accents and arrangement changes "
            "to give different sections more contrast."
        )
    else:
        suggestions.append(
            f"The estimated tempo is {bpm} BPM, which is relatively fast. "
            "Try simplifying busy drum patterns or using contrasting "
            "slower sections if the track feels too intense."
        )

    if duration > 0:
        beats_per_second = beat_count / duration

        if beats_per_second < 0.8:
            suggestions.append(
                "Few beats were detected relative to the track duration. "
                "Check the timestamps manually; the beat detector may have "
                "missed beats in quieter or less rhythmic sections."
            )
        elif beats_per_second > 2.5:
            suggestions.append(
                "Many beats were detected per second. Check the timestamps "
                "against playback to see whether subdivisions were detected "
                "instead of the main beats."
            )
        else:
            suggestions.append(
                f"The detector found about {beats_per_second:.2f} beats "
                "per second on average. Compare the timestamps with the "
                "audio to assess beat-detection accuracy."
            )

    if duration < 60:
        suggestions.append(
            "This is a short track or audio excerpt. Analyse a longer "
            "section to assess how the rhythm behaves across the song."
        )
    elif duration >= 180:
        suggestions.append(
            "This is a longer track. Compare different sections separately "
            "to identify possible changes in rhythm and arrangement."
        )

    suggestions.append(
        "These suggestions use estimated tempo, beat density and duration. "
        "They are rule-based and do not come from a trained ML model."
    )

    return suggestions




def analyse_audio(filepath):
    y, sr = librosa.load(filepath, sr=22050, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio file")

    # Tempo and beat detection
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    bpm = float(np.asarray(tempo).reshape(-1)[0])
    duration = round(len(y) / sr, 2)
    bpm_rounded = round(bpm, 1)
    beat_count = len(beat_times)

    # Audio features
    rms = librosa.feature.rms(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr
    )[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

    rms_mean = float(np.mean(rms))
    centroid_mean = float(np.mean(spectral_centroid))
    zcr_mean = float(np.mean(zero_crossing_rate))

    # Basic interpretations of the measured features
    if rms_mean < 0.05:
        energy_description = "Low average signal energy"
    elif rms_mean < 0.15:
        energy_description = "Moderate average signal energy"
    else:
        energy_description = "High average signal energy"

    if centroid_mean < 1500:
        brightness_description = "Darker spectral character"
    elif centroid_mean < 3000:
        brightness_description = "Moderate spectral brightness"
    else:
        brightness_description = "Brighter spectral character"

    return {
        "bpm": bpm_rounded,
        "duration": duration,
        "beat_count": beat_count,
        "timestamps": ", ".join(
            f"{time:.2f}" for time in beat_times[:200]
        ) or "No beats detected",
        "beat_times": [
            round(float(t), 3) for t in beat_times
        ],
        "rms_energy": round(rms_mean, 4),
        "spectral_centroid_hz": round(centroid_mean, 2),
        "zero_crossing_rate": round(zcr_mean, 4),
        "energy_description": energy_description,
        "brightness_description": brightness_description,
        "suggestions": generate_suggestions(
            bpm_rounded, beat_count, duration
        ),
    }



@app.route("/", methods=["GET", "POST"])
def home():
    message = None
    audio_url = None
    audio_type = None
    results = None

    if request.method == "POST":
        file = request.files.get("audio")

        if not file or not file.filename:
            message = "Please select a song."

        elif not allowed_file(file.filename):
            message = "Only MP3 and WAV files are supported."

        else:
            filename = secure_filename(file.filename)

            if not filename:
                message = "Invalid filename."

            else:
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                try:
                    file.save(filepath)
                    results = analyse_audio(filepath)

                    message = "Song uploaded and analysed successfully!"
                    audio_url = f"/uploads/{filename}"

                    audio_type = (
                        "audio/mpeg"
                        if filename.lower().endswith(".mp3")
                        else "audio/wav"
                    )

                except Exception:
                    app.logger.exception("Audio analysis failed")

                    if os.path.exists(filepath):
                        os.remove(filepath)

                    message = (
                        "Could not analyse this file. "
                        "Try another valid MP3 or WAV file."
                    )

    return render_template_string(
        HTML,
        message=message,
        audio_url=audio_url,
        audio_type=audio_type,
        results=results,
        song_name=filename if audio_url else None,
    )


@app.route("/uploads/<path:filename>")
def uploaded_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.errorhandler(413)
def file_too_large(error):
    return "File too large. Maximum size is 20 MB.", 413


if __name__ == "__main__":
    app.run(debug=True)
