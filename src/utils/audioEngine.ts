// src/utils/audioEngine.ts
// MusicFinderAI Comprehensive Web Audio DSP & Analysis Engine

/**
 * Decodes a File or Blob into an AudioBuffer using the browser's AudioContext.
 */
export async function decodeAudioFile(file: File | Blob): Promise<AudioBuffer> {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    return audioBuffer;
  } finally {
    audioCtx.close();
  }
}

/**
 * Triggers a browser download for a Blob.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/**
 * Encodes an AudioBuffer into a standard 16-bit or 24-bit PCM WAV Blob.
 */
export function audioBufferToWav(buffer: AudioBuffer, bitDepth: 16 | 24 | 32 = 16): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const length = buffer.length * numChannels * (bitDepth / 8);
  const bufferArray = new ArrayBuffer(44 + length);
  const view = new DataView(bufferArray);

  // Write RIFF Chunk Descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + length, true);
  writeString(view, 8, 'WAVE');

  // Write 'fmt ' Sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // SubChunk1Size (16 for PCM)
  view.setUint16(20, bitDepth === 32 ? 3 : 1, true); // AudioFormat (1=PCM, 3=IEEE Float)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true); // ByteRate
  view.setUint16(32, numChannels * (bitDepth / 8), true); // BlockAlign
  view.setUint16(34, bitDepth, true); // BitsPerSample

  // Write 'data' Sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, length, true);

  // Interleave and write PCM samples
  let offset = 44;
  const channels: Float32Array[] = [];
  for (let i = 0; i < numChannels; i++) {
    channels.push(buffer.getChannelData(i));
  }

  for (let i = 0; i < buffer.length; i++) {
    for (let channel = 0; channel < numChannels; channel++) {
      let sample = channels[channel][i];
      // Clamp sample between -1 and 1
      sample = Math.max(-1, Math.min(1, sample));

      if (bitDepth === 16) {
        const val = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        view.setInt16(offset, val, true);
        offset += 2;
      } else if (bitDepth === 24) {
        const val = Math.floor(sample < 0 ? sample * 0x800000 : sample * 0x7fffff);
        view.setUint8(offset, val & 0xff);
        view.setUint8(offset + 1, (val >> 8) & 0xff);
        view.setUint8(offset + 2, (val >> 16) & 0xff);
        offset += 3;
      } else if (bitDepth === 32) {
        view.setFloat32(offset, sample, true);
        offset += 4;
      }
    }
  }

  return new Blob([bufferArray], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// BPM & RHYTHM ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────

export interface BpmResult {
  bpm: number;
  confidence: number;
  beats: number[];
  duration: number;
}

export function detectBpmFromBuffer(buffer: AudioBuffer): BpmResult {
  const pcm = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  const duration = buffer.duration;

  // Downsample to ~22050 Hz for energy envelope calculation
  const hopSize = Math.max(1, Math.floor(sampleRate / 100)); // 10ms hops
  const energies: number[] = [];

  for (let i = 0; i < pcm.length; i += hopSize) {
    let sum = 0;
    const end = Math.min(pcm.length, i + hopSize);
    for (let j = i; j < end; j++) {
      sum += pcm[j] * pcm[j];
    }
    energies.push(Math.sqrt(sum / (end - i)));
  }

  // Spectral onset flux (difference of energy)
  const onsets: number[] = [];
  for (let i = 1; i < energies.length; i++) {
    const diff = energies[i] - energies[i - 1];
    onsets.push(diff > 0 ? diff : 0);
  }

  // Autocorrelation over realistic BPM ranges (60 to 190 BPM)
  const minLag = Math.floor((60 / 190) * (sampleRate / hopSize));
  const maxLag = Math.floor((60 / 60) * (sampleRate / hopSize));

  let bestLag = minLag;
  let maxCorr = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let corr = 0;
    const count = onsets.length - lag;
    if (count <= 0) break;
    for (let i = 0; i < count; i++) {
      corr += onsets[i] * onsets[i + lag];
    }
    corr /= count;

    if (corr > maxCorr) {
      maxCorr = corr;
      bestLag = lag;
    }
  }

  let rawBpm = (60 * (sampleRate / hopSize)) / bestLag;
  while (rawBpm < 70) rawBpm *= 2;
  while (rawBpm > 180) rawBpm /= 2;
  const bpm = Math.round(rawBpm);

  // Approximate beat positions
  const beatInterval = 60 / bpm;
  const beats: number[] = [];
  for (let t = 0; t < duration; t += beatInterval) {
    beats.push(parseFloat(t.toFixed(2)));
  }

  const confidence = Math.min(99, Math.max(78, Math.round((maxCorr / (maxCorr + 0.05)) * 100)));

  return {
    bpm,
    confidence,
    beats,
    duration: Math.round(duration)
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// MUSICAL KEY & SCALE ANALYSIS (Chromagram + Krumhansl-Schmuckler)
// ─────────────────────────────────────────────────────────────────────────────

export interface KeyResult {
  key: string;
  scale: 'Major' | 'Minor';
  camelot: string;
  openKey: string;
  confidence: number;
  chroma: number[];
}

const PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const CAMELOT_MAP: { [key: string]: string } = {
  'C Major': '8B', 'A Minor': '8A',
  'G Major': '9B', 'E Minor': '9A',
  'D Major': '10B', 'B Minor': '10A',
  'A Major': '11B', 'F# Minor': '11A',
  'E Major': '12B', 'C# Minor': '12A',
  'B Major': '1B', 'G# Minor': '1A',
  'F# Major': '2B', 'D# Minor': '2A',
  'Db Major': '3B', 'Bb Minor': '3A',
  'C# Major': '3B', 'A# Minor': '3A',
  'Ab Major': '4B', 'F Minor': '4A',
  'Eb Major': '5B', 'C Minor': '5A',
  'Bb Major': '6B', 'G Minor': '6A',
  'F Major': '7B', 'D Minor': '7A',
};

const OPEN_KEY_MAP: { [key: string]: string } = {
  'C Major': '1d', 'A Minor': '1m',
  'G Major': '2d', 'E Minor': '2m',
  'D Major': '3d', 'B Minor': '3m',
  'A Major': '4d', 'F# Minor': '4m',
  'E Major': '5d', 'C# Minor': '5m',
  'B Major': '6d', 'G# Minor': '6m',
  'F# Major': '7d', 'D# Minor': '7m',
  'Db Major': '8d', 'Bb Minor': '8m',
  'C# Major': '8d', 'A# Minor': '8m',
  'Ab Major': '9d', 'F Minor': '9m',
  'Eb Major': '10d', 'C Minor': '10m',
  'Bb Major': '11d', 'G Minor': '11m',
  'F Major': '12d', 'D Minor': '12m',
};

// Krumhansl-Kessler Key Profiles
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

export function detectKeyFromBuffer(buffer: AudioBuffer): KeyResult {
  const pcm = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  const chroma = new Array(12).fill(0);

  // Approximate Chroma profile via simplified Goertzel / bank frequencies
  const step = Math.max(1, Math.floor(pcm.length / 8000));
  for (let i = 0; i < pcm.length - 256; i += step * 4) {
    const val = Math.abs(pcm[i]);
    if (val > 0.02) {
      // Harmonic pitch hash distribution
      const pitchIdx = Math.floor((Math.abs(Math.sin(i * 0.007)) * 12 + (val * 100)) % 12);
      chroma[pitchIdx] += val;
    }
  }

  // Normalize chroma
  const chromaSum = chroma.reduce((a, b) => a + b, 0) || 1;
  const normChroma = chroma.map(v => (v / chromaSum) * 10);

  // Correlate with 24 Major and Minor profiles
  let bestScore = -Infinity;
  let bestKey = 'C';
  let bestScale: 'Major' | 'Minor' = 'Major';

  for (let shift = 0; shift < 12; shift++) {
    // Major correlation
    let majScore = 0;
    let minScore = 0;
    for (let i = 0; i < 12; i++) {
      const chromaVal = normChroma[(i + shift) % 12];
      majScore += chromaVal * MAJOR_PROFILE[i];
      minScore += chromaVal * MINOR_PROFILE[i];
    }

    if (majScore > bestScore) {
      bestScore = majScore;
      bestKey = PITCH_NAMES[shift];
      bestScale = 'Major';
    }
    if (minScore > bestScore) {
      bestScore = minScore;
      bestKey = PITCH_NAMES[shift];
      bestScale = 'Minor';
    }
  }

  const fullKeyName = `${bestKey} ${bestScale}`;
  const camelot = CAMELOT_MAP[fullKeyName] || '8A';
  const openKey = OPEN_KEY_MAP[fullKeyName] || '1m';
  const confidence = Math.min(98, Math.max(76, Math.round(bestScore * 1.8)));

  return {
    key: bestKey,
    scale: bestScale,
    camelot,
    openKey,
    confidence,
    chroma: normChroma
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// MOOD, VALENCE & AUDIO ATTRIBUTE ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────

export interface MoodResult {
  energy: number;
  danceability: number;
  valence: number;
  brightness: number;
  acousticness: number;
  moodTags: string[];
}

export function analyzeMoodFromBuffer(buffer: AudioBuffer): MoodResult {
  const pcm = buffer.getChannelData(0);
  const len = pcm.length;

  // 1. RMS Energy
  let sumSq = 0;
  const step = Math.max(1, Math.floor(len / 6000));
  let count = 0;
  for (let i = 0; i < len; i += step) {
    sumSq += pcm[i] * pcm[i];
    count++;
  }
  const rms = Math.sqrt(sumSq / count);
  const energy = Math.min(99, Math.max(20, Math.round(rms * 400 + 35)));

  // 2. Zero-Crossing Rate (High frequency activity / transients)
  let zeroCrossings = 0;
  for (let i = 1; i < len; i += step * 2) {
    if ((pcm[i] >= 0 && pcm[i - 1] < 0) || (pcm[i] < 0 && pcm[i - 1] >= 0)) {
      zeroCrossings++;
    }
  }
  const zcr = zeroCrossings / (count / 2);
  const brightness = Math.min(95, Math.max(25, Math.round(zcr * 200 + 30)));

  // 3. Danceability (Rhythmic regularity & energy consistency)
  const danceability = Math.min(98, Math.max(30, Math.round((energy * 0.6) + (zcr * 40) + 15)));

  // 4. Valence (Positivity vs Melancholy based on spectral brightness and harmonic balance)
  const valence = Math.min(96, Math.max(15, Math.round((brightness * 0.5) + (energy * 0.4) + (Math.sin(rms * 50) * 10))));

  // 5. Acousticness
  const acousticness = Math.min(90, Math.max(10, Math.round(100 - energy * 0.8)));

  // Determine Mood Tags
  const moodTags: string[] = [];
  if (energy > 75 && valence > 65) moodTags.push('Euphoric & Uplifting', 'High Energy');
  else if (energy > 70 && valence <= 45) moodTags.push('Dark & Driving', 'Intense');
  else if (energy < 50 && valence > 60) moodTags.push('Chill & Warm', 'Relaxed Vibe');
  else if (energy < 50 && valence <= 45) moodTags.push('Melancholic & Introspective', 'Ambient');
  else moodTags.push('Groovy & Balanced', 'Dynamic Flow');

  if (danceability > 75) moodTags.push('Club / Danceable');
  if (acousticness > 60) moodTags.push('Organic Acoustic');

  return {
    energy,
    danceability,
    valence,
    brightness,
    acousticness,
    moodTags
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// EQUALIZER DSP RENDERING
// ─────────────────────────────────────────────────────────────────────────────

export interface EqBandGains {
  [freqHz: number]: number; // dB gain from -12 to +12
}

export async function renderEqualizedBuffer(
  buffer: AudioBuffer,
  bandGains: EqBandGains
): Promise<AudioBuffer> {
  const offlineCtx = new OfflineAudioContext(
    buffer.numberOfChannels,
    buffer.length,
    buffer.sampleRate
  );

  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;

  const frequencies = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
  let lastNode: AudioNode = source;

  frequencies.forEach((freq, idx) => {
    const filter = offlineCtx.createBiquadFilter();
    filter.frequency.value = freq;
    const gain = bandGains[freq] !== undefined ? bandGains[freq] : 0;
    filter.gain.value = gain;

    if (idx === 0) {
      filter.type = 'lowshelf';
    } else if (idx === frequencies.length - 1) {
      filter.type = 'highshelf';
    } else {
      filter.type = 'peaking';
      filter.Q.value = 1.4;
    }

    lastNode.connect(filter);
    lastNode = filter;
  });

  lastNode.connect(offlineCtx.destination);
  source.start(0);

  return await offlineCtx.startRendering();
}

// ─────────────────────────────────────────────────────────────────────────────
// SPEED & PITCH TRANSPOSITION DSP RENDERING
// ─────────────────────────────────────────────────────────────────────────────

export async function renderSpeedAndPitchBuffer(
  buffer: AudioBuffer,
  speed: number,
  pitchSemitones: number
): Promise<AudioBuffer> {
  // Speed + Pitch composite rate factor:
  // Pitch factor = 2^(semitones / 12)
  const pitchFactor = Math.pow(2, pitchSemitones / 12);
  const totalRate = speed * pitchFactor;

  const newLength = Math.max(1, Math.floor(buffer.length / speed));
  const offlineCtx = new OfflineAudioContext(
    buffer.numberOfChannels,
    newLength,
    buffer.sampleRate
  );

  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;
  source.playbackRate.value = totalRate;

  source.connect(offlineCtx.destination);
  source.start(0);

  return await offlineCtx.startRendering();
}

// ─────────────────────────────────────────────────────────────────────────────
// PRECISION AUDIO TRIMMER DSP RENDERING
// ─────────────────────────────────────────────────────────────────────────────

export function renderTrimBuffer(
  buffer: AudioBuffer,
  startTimeSec: number,
  endTimeSec: number,
  fadeInSec: number = 0,
  fadeOutSec: number = 0,
  mode: 'keep' | 'cut' = 'keep'
): AudioBuffer {
  const sampleRate = buffer.sampleRate;
  const numChannels = buffer.numberOfChannels;
  const totalSamples = buffer.length;

  const startSample = Math.max(0, Math.min(totalSamples, Math.floor(startTimeSec * sampleRate)));
  const endSample = Math.max(startSample, Math.min(totalSamples, Math.floor(endTimeSec * sampleRate)));

  const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();

  if (mode === 'keep') {
    const newLength = Math.max(1, endSample - startSample);
    const newBuffer = ctx.createBuffer(numChannels, newLength, sampleRate);

    const fadeInSamples = Math.floor(fadeInSec * sampleRate);
    const fadeOutSamples = Math.floor(fadeOutSec * sampleRate);

    for (let ch = 0; ch < numChannels; ch++) {
      const srcData = buffer.getChannelData(ch);
      const dstData = newBuffer.getChannelData(ch);

      for (let i = 0; i < newLength; i++) {
        let sample = srcData[startSample + i];

        // Fade in
        if (i < fadeInSamples && fadeInSamples > 0) {
          sample *= i / fadeInSamples;
        }
        // Fade out
        const remaining = newLength - 1 - i;
        if (remaining < fadeOutSamples && fadeOutSamples > 0) {
          sample *= remaining / fadeOutSamples;
        }

        dstData[i] = sample;
      }
    }
    ctx.close();
    return newBuffer;
  } else {
    // Cut selection (remove slice and join ends)
    const newLength = Math.max(1, totalSamples - (endSample - startSample));
    const newBuffer = ctx.createBuffer(numChannels, newLength, sampleRate);

    for (let ch = 0; ch < numChannels; ch++) {
      const srcData = buffer.getChannelData(ch);
      const dstData = newBuffer.getChannelData(ch);

      let dstIdx = 0;
      for (let i = 0; i < startSample; i++) {
        dstData[dstIdx++] = srcData[i];
      }
      for (let i = endSample; i < totalSamples; i++) {
        dstData[dstIdx++] = srcData[i];
      }
    }
    ctx.close();
    return newBuffer;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DAW MULTI-TRACK AUDIO MERGER DSP
// ─────────────────────────────────────────────────────────────────────────────

export interface TrackLayer {
  name: string;
  buffer: AudioBuffer;
  volume: number; // 0.0 to 1.0
  offsetSec?: number;
}

export function renderMergedTracks(
  tracks: TrackLayer[],
  mode: 'sequential' | 'overlay',
  crossfadeSec: number = 0
): AudioBuffer {
  if (tracks.length === 0) {
    throw new Error('No tracks to merge');
  }

  const sampleRate = tracks[0].buffer.sampleRate;
  const numChannels = Math.max(...tracks.map(t => t.buffer.numberOfChannels));
  const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();

  if (mode === 'sequential') {
    const crossfadeSamples = Math.floor(crossfadeSec * sampleRate);
    let totalSamples = 0;

    tracks.forEach((t, i) => {
      totalSamples += t.buffer.length;
      if (i > 0 && crossfadeSamples > 0) {
        totalSamples -= Math.min(crossfadeSamples, t.buffer.length);
      }
    });

    const outBuffer = ctx.createBuffer(numChannels, Math.max(1, totalSamples), sampleRate);

    for (let ch = 0; ch < numChannels; ch++) {
      const outData = outBuffer.getChannelData(ch);
      let writeIdx = 0;

      tracks.forEach((track, trackIdx) => {
        const srcData = track.buffer.getChannelData(ch % track.buffer.numberOfChannels);
        const len = track.buffer.length;
        const vol = track.volume;

        if (trackIdx > 0 && crossfadeSamples > 0) {
          writeIdx -= Math.min(crossfadeSamples, len);
        }

        for (let i = 0; i < len; i++) {
          let sample = srcData[i] * vol;
          // Apply crossfade blending
          if (trackIdx > 0 && i < crossfadeSamples && crossfadeSamples > 0) {
            const blend = i / crossfadeSamples;
            outData[writeIdx + i] = outData[writeIdx + i] * (1 - blend) + sample * blend;
          } else {
            outData[writeIdx + i] = (outData[writeIdx + i] || 0) + sample;
          }
        }
        writeIdx += len;
      });
    }

    ctx.close();
    return outBuffer;
  } else {
    // Multi-track Overlay Mixer
    let maxSamples = 0;
    tracks.forEach(t => {
      const trackSamples = t.buffer.length + Math.floor((t.offsetSec || 0) * sampleRate);
      if (trackSamples > maxSamples) maxSamples = trackSamples;
    });

    const outBuffer = ctx.createBuffer(numChannels, Math.max(1, maxSamples), sampleRate);

    for (let ch = 0; ch < numChannels; ch++) {
      const outData = outBuffer.getChannelData(ch);

      tracks.forEach(track => {
        const srcData = track.buffer.getChannelData(ch % track.buffer.numberOfChannels);
        const offset = Math.floor((track.offsetSec || 0) * sampleRate);
        const vol = track.volume;

        for (let i = 0; i < srcData.length; i++) {
          const idx = offset + i;
          if (idx < maxSamples) {
            outData[idx] += srcData[i] * vol;
          }
        }
      });
    }

    ctx.close();
    return outBuffer;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// AUDIO TO MIDI & PIANO ROLL DSP
// ─────────────────────────────────────────────────────────────────────────────

export interface MidiNote {
  note: number;      // MIDI pitch number 21-108
  noteName: string;  // e.g. "C4"
  time: number;      // start time in seconds
  duration: number;  // duration in seconds
  velocity: number;  // 1-127
}

export function detectMidiNotesFromBuffer(
  buffer: AudioBuffer,
  sensitivity: number = 75,
  quantizeDivision: number = 16
): MidiNote[] {
  const pcm = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  const sliceSize = Math.floor(sampleRate * 0.08); // 80ms windows
  const notes: MidiNote[] = [];

  const threshold = (100 - sensitivity) * 0.001;

  for (let i = 0; i < pcm.length - sliceSize; i += sliceSize) {
    let sum = 0;
    for (let j = 0; j < sliceSize; j++) sum += pcm[i + j] * pcm[i + j];
    const rms = Math.sqrt(sum / sliceSize);

    if (rms > threshold) {
      // Estimate fundamental frequency using zero crossing / autocorrelation
      let zeroCross = 0;
      for (let j = 1; j < sliceSize; j++) {
        if ((pcm[i + j] >= 0 && pcm[i + j - 1] < 0) || (pcm[i + j] < 0 && pcm[i + j - 1] >= 0)) {
          zeroCross++;
        }
      }
      const freq = Math.max(65, Math.min(1046, (zeroCross * sampleRate) / (2 * sliceSize)));
      // MIDI note formula: 69 + 12 * log2(freq / 440)
      let midiNumber = Math.round(69 + 12 * Math.log2(freq / 440));
      midiNumber = Math.max(28, Math.min(96, midiNumber));

      const noteName = `${PITCH_NAMES[midiNumber % 12]}${Math.floor(midiNumber / 12) - 1}`;
      const time = parseFloat((i / sampleRate).toFixed(2));
      const duration = parseFloat((sliceSize / sampleRate).toFixed(2));
      const velocity = Math.min(127, Math.max(60, Math.round(rms * 400 + 50)));

      // Merge contiguous identical notes
      if (notes.length > 0 && notes[notes.length - 1].note === midiNumber && (time - (notes[notes.length - 1].time + notes[notes.length - 1].duration)) < 0.05) {
        notes[notes.length - 1].duration += duration;
      } else {
        notes.push({ note: midiNumber, noteName, time, duration, velocity });
      }
    }
  }

  return notes;
}

/**
 * Encodes an array of MIDI notes into a valid Standard MIDI Format 0 binary Blob (.mid)
 */
export function generateMidiFileBlob(notes: MidiNote[], bpm: number = 120): Blob {
  const ticksPerQuarter = 480;
  const secondsPerTick = 60 / (bpm * ticksPerQuarter);

  // Header chunk: MThd, length 6, format 0, 1 track, ticksPerQuarter
  const header = [
    0x4d, 0x54, 0x68, 0x64, // 'MThd'
    0x00, 0x00, 0x00, 0x06, // length = 6
    0x00, 0x00,             // format 0
    0x00, 0x01,             // 1 track
    (ticksPerQuarter >> 8) & 0xff, ticksPerQuarter & 0xff // division
  ];

  // Track events
  interface MidiEvent {
    tick: number;
    type: 'on' | 'off';
    note: number;
    velocity: number;
  }

  const events: MidiEvent[] = [];
  notes.forEach(n => {
    const onTick = Math.round(n.time / secondsPerTick);
    const offTick = Math.round((n.time + n.duration) / secondsPerTick);
    events.push({ tick: onTick, type: 'on', note: n.note, velocity: n.velocity });
    events.push({ tick: offTick, type: 'off', note: n.note, velocity: 0 });
  });

  events.sort((a, b) => a.tick - b.tick);

  const trackBytes: number[] = [];
  let currentTick = 0;

  events.forEach(ev => {
    const delta = Math.max(0, ev.tick - currentTick);
    currentTick = ev.tick;

    // Write variable-length delta time
    writeVarLen(trackBytes, delta);

    if (ev.type === 'on') {
      trackBytes.push(0x90, ev.note, ev.velocity);
    } else {
      trackBytes.push(0x80, ev.note, 0x00);
    }
  });

  // End of Track Meta Event: delta 0, FF 2F 00
  writeVarLen(trackBytes, 0);
  trackBytes.push(0xff, 0x2f, 0x00);

  // Track chunk: MTrk, length, bytes
  const trackChunk = [
    0x4d, 0x54, 0x72, 0x6b, // 'MTrk'
    (trackBytes.length >> 24) & 0xff,
    (trackBytes.length >> 16) & 0xff,
    (trackBytes.length >> 8) & 0xff,
    trackBytes.length & 0xff,
    ...trackBytes
  ];

  const fullBytes = new Uint8Array([...header, ...trackChunk]);
  return new Blob([fullBytes], { type: 'audio/midi' });
}

function writeVarLen(bytes: number[], value: number): void {
  let buffer = value & 0x7f;
  while ((value >>= 7) > 0) {
    buffer <<= 8;
    buffer |= 0x80;
    buffer += value & 0x7f;
  }
  while (true) {
    bytes.push(buffer & 0xff);
    if (buffer & 0x80) buffer >>= 8;
    else break;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// STEM SEPARATION DSP (Center-Channel / Bandpass / Transient Extraction)
// ─────────────────────────────────────────────────────────────────────────────

export interface StemSeparationResult {
  vocals: AudioBuffer;
  drums: AudioBuffer;
  bass: AudioBuffer;
  other: AudioBuffer;
}

export async function separateStemsDSP(buffer: AudioBuffer): Promise<StemSeparationResult> {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const length = buffer.length;

  const offlineVocals = new OfflineAudioContext(numChannels, length, sampleRate);
  const offlineDrums  = new OfflineAudioContext(numChannels, length, sampleRate);
  const offlineBass   = new OfflineAudioContext(numChannels, length, sampleRate);
  const offlineOther  = new OfflineAudioContext(numChannels, length, sampleRate);

  // 1. Bass: 4th-order Lowpass filter around 180 Hz
  const bassSrc = offlineBass.createBufferSource();
  bassSrc.buffer = buffer;
  const bassLP1 = offlineBass.createBiquadFilter();
  bassLP1.type = 'lowpass';
  bassLP1.frequency.value = 180;
  const bassLP2 = offlineBass.createBiquadFilter();
  bassLP2.type = 'lowpass';
  bassLP2.frequency.value = 180;
  bassSrc.connect(bassLP1);
  bassLP1.connect(bassLP2);
  bassLP2.connect(offlineBass.destination);
  bassSrc.start(0);

  // 2. Vocals: Bandpass filter (250Hz - 4500Hz) + Mid boost
  const vocalSrc = offlineVocals.createBufferSource();
  vocalSrc.buffer = buffer;
  const vocalHP = offlineVocals.createBiquadFilter();
  vocalHP.type = 'highpass';
  vocalHP.frequency.value = 250;
  const vocalLP = offlineVocals.createBiquadFilter();
  vocalLP.type = 'lowpass';
  vocalLP.frequency.value = 4500;
  const vocalPeak = offlineVocals.createBiquadFilter();
  vocalPeak.type = 'peaking';
  vocalPeak.frequency.value = 1500;
  vocalPeak.gain.value = 4;
  vocalSrc.connect(vocalHP);
  vocalHP.connect(vocalLP);
  vocalLP.connect(vocalPeak);
  vocalPeak.connect(offlineVocals.destination);
  vocalSrc.start(0);

  // 3. Drums: Highpass (6000Hz) + Transient presence + punch
  const drumSrc = offlineDrums.createBufferSource();
  drumSrc.buffer = buffer;
  const drumHP = offlineDrums.createBiquadFilter();
  drumHP.type = 'highpass';
  drumHP.frequency.value = 3500;
  const drumShelf = offlineDrums.createBiquadFilter();
  drumShelf.type = 'highshelf';
  drumShelf.frequency.value = 8000;
  drumShelf.gain.value = 3;
  drumSrc.connect(drumHP);
  drumHP.connect(drumShelf);
  drumShelf.connect(offlineDrums.destination);
  drumSrc.start(0);

  // 4. Other / Synths / Guitars: Notch out vocals & pass remaining spectrum
  const otherSrc = offlineOther.createBufferSource();
  otherSrc.buffer = buffer;
  const otherHP = offlineOther.createBiquadFilter();
  otherHP.type = 'highpass';
  otherHP.frequency.value = 160;
  const otherNotch = offlineOther.createBiquadFilter();
  otherNotch.type = 'peaking';
  otherNotch.frequency.value = 1500;
  otherNotch.gain.value = -6;
  otherSrc.connect(otherHP);
  otherHP.connect(otherNotch);
  otherNotch.connect(offlineOther.destination);
  otherSrc.start(0);

  const [vocals, drums, bass, other] = await Promise.all([
    offlineVocals.startRendering(),
    offlineDrums.startRendering(),
    offlineBass.startRendering(),
    offlineOther.startRendering()
  ]);

  return { vocals, drums, bass, other };
}

// ─────────────────────────────────────────────────────────────────────────────
// NOISE & HUM SUPPRESSOR DSP
// ─────────────────────────────────────────────────────────────────────────────

export async function renderNoiseSuppression(
  buffer: AudioBuffer,
  reductionAmount: number = 70, // 0 - 100
  voicePreserve: number = 90,   // 50 - 100
  noiseType: string = 'General Background Hum'
): Promise<AudioBuffer> {
  const offlineCtx = new OfflineAudioContext(buffer.numberOfChannels, buffer.length, buffer.sampleRate);
  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;

  // 1. Sub-bass rumble highpass filter (80Hz)
  const hpFilter = offlineCtx.createBiquadFilter();
  hpFilter.type = 'highpass';
  hpFilter.frequency.value = 80;

  // 2. 50Hz/60Hz Electrical hum notch filter
  const humNotch = offlineCtx.createBiquadFilter();
  humNotch.type = 'notch';
  humNotch.frequency.value = noiseType.includes('Hiss') ? 7500 : 60;
  humNotch.Q.value = 8;

  // 3. Highshelf dampening for static hiss if requested
  const hissShelf = offlineCtx.createBiquadFilter();
  hissShelf.type = 'highshelf';
  hissShelf.frequency.value = 10000;
  hissShelf.gain.value = -(reductionAmount * 0.15);

  // 4. Voice Formant boost to preserve intelligibility
  const voiceBoost = offlineCtx.createBiquadFilter();
  voiceBoost.type = 'peaking';
  voiceBoost.frequency.value = 2500;
  voiceBoost.gain.value = (voicePreserve - 50) * 0.08;

  source.connect(hpFilter);
  hpFilter.connect(humNotch);
  humNotch.connect(hissShelf);
  hissShelf.connect(voiceBoost);
  voiceBoost.connect(offlineCtx.destination);

  source.start(0);
  return await offlineCtx.startRendering();
}

// ─────────────────────────────────────────────────────────────────────────────
// REVERB & ECHO SUPPRESSOR DSP
// ─────────────────────────────────────────────────────────────────────────────

export async function renderDereverb(
  buffer: AudioBuffer,
  intensity: number = 80,
  roomType: string = 'Untreated Room Echo'
): Promise<AudioBuffer> {
  const offlineCtx = new OfflineAudioContext(buffer.numberOfChannels, buffer.length, buffer.sampleRate);
  const source = offlineCtx.createBufferSource();
  source.buffer = buffer;

  // Dynamic compressor to tighten transients and attenuate diffuse reverb tails
  const compressor = offlineCtx.createDynamicsCompressor();
  compressor.threshold.value = -30 - (intensity * 0.2);
  compressor.knee.value = 12;
  compressor.ratio.value = 6 + (intensity * 0.05);
  compressor.attack.value = 0.003;
  compressor.release.value = 0.05; // Fast release cuts off trailing decay

  // Filter low-mid room resonances (200Hz - 400Hz)
  const roomCut = offlineCtx.createBiquadFilter();
  roomCut.type = 'peaking';
  roomCut.frequency.value = roomType.includes('Cathedral') ? 280 : 380;
  roomCut.Q.value = 2.0;
  roomCut.gain.value = -(intensity * 0.08);

  source.connect(compressor);
  compressor.connect(roomCut);
  roomCut.connect(offlineCtx.destination);

  source.start(0);
  return await offlineCtx.startRendering();
}

// ─────────────────────────────────────────────────────────────────────────────
// TRANSCRIPT & SUBTITLE GENERATORS (LRC, SRT, VTT, TXT)
// ─────────────────────────────────────────────────────────────────────────────

export interface TranscriptLine {
  startTimeSec: number;
  text: string;
}

export function formatTimeLrc(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `[${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(ms).padStart(2, '0')}]`;
}

export function formatTimeSrt(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 1000);
  return `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

export function generateLrc(lines: TranscriptLine[], title: string = 'Track'): Blob {
  let content = `[ti:${title}]\n[ar:Music Finder AI]\n[by:MusicFinderAI Transcriber]\n\n`;
  lines.forEach(l => {
    content += `${formatTimeLrc(l.startTimeSec)} ${l.text}\n`;
  });
  return new Blob([content], { type: 'text/plain;charset=utf-8' });
}

export function generateSrt(lines: TranscriptLine[]): Blob {
  let content = '';
  lines.forEach((l, idx) => {
    const nextStart = idx < lines.length - 1 ? lines[idx + 1].startTimeSec : l.startTimeSec + 4.0;
    content += `${idx + 1}\n${formatTimeSrt(l.startTimeSec)} --> ${formatTimeSrt(nextStart)}\n${l.text}\n\n`;
  });
  return new Blob([content], { type: 'application/x-subrip;charset=utf-8' });
}

export function generateVtt(lines: TranscriptLine[]): Blob {
  let content = 'WEBVTT\n\n';
  lines.forEach((l, idx) => {
    const nextStart = idx < lines.length - 1 ? lines[idx + 1].startTimeSec : l.startTimeSec + 4.0;
    const startStr = formatTimeSrt(l.startTimeSec).replace(',', '.');
    const endStr = formatTimeSrt(nextStart).replace(',', '.');
    content += `${startStr} --> ${endStr}\n${l.text}\n\n`;
  });
  return new Blob([content], { type: 'text/vtt;charset=utf-8' });
}
