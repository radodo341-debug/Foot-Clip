"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_PITCHPERFECT_API ?? "http://127.0.0.1:8000";

type Job = { id: string; status: string; progress: number; stage: string; error?: string | null; duration_seconds?: number };
type Track = { track_id: number; first_seen_seconds: number; last_seen_seconds: number; observations: number; player_name?: string | null; role?: string | null; team_name?: string | null };
type Clip = { id: string; title: string; category: string; start_seconds: number; end_seconds: number; status: string; confidence?: number | null; notes?: string | null };

export default function PitchPerfectUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [matchName, setMatchName] = useState("Real Madrid vs Barcelona");
  const [job, setJob] = useState<Job | null>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [clips, setClips] = useState<Clip[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<number | null>(null);
  const [playerName, setPlayerName] = useState("Kylian Mbappé");
  const [teamName, setTeamName] = useState("Real Madrid");
  const [start, setStart] = useState("0");
  const [end, setEnd] = useState("20");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function api<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API}${path}`, init);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail ?? `Request failed (${response.status})`);
    }
    return response.json();
  }

  async function uploadMatch() {
    if (!file) return setMessage("Choose a match video first.");
    setBusy(true); setMessage("Uploading match…");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("match_name", matchName);
      const result = await api<{ job_id: string; status: string; duration_seconds: number }>("/api/matches/upload", { method: "POST", body: form });
      setJob({ id: result.job_id, status: result.status, progress: 0, stage: "queued", duration_seconds: result.duration_seconds });
      setTracks([]); setClips([]); setMessage("Upload accepted. Analysis is running.");
    } catch (e) { setMessage(e instanceof Error ? e.message : "Upload failed."); }
    finally { setBusy(false); }
  }

  useEffect(() => {
    if (!job || ["completed", "failed"].includes(job.status)) return;
    const timer = setInterval(async () => {
      try {
        const next = await api<Job>(`/api/jobs/${job.id}`);
        setJob(next);
        if (next.status === "completed") {
          const [trackData, clipData] = await Promise.all([
            api<{ tracks: Track[] }>(`/api/jobs/${job.id}/tracks`),
            api<{ clips: Clip[] }>(`/api/jobs/${job.id}/clips`),
          ]);
          setTracks(trackData.tracks); setClips(clipData.clips);
          setMessage("Analysis finished. Confirm a track identity, then create clips.");
        }
        if (next.status === "failed") setMessage(next.error ?? "Analysis failed.");
      } catch (e) { setMessage(e instanceof Error ? e.message : "Status check failed."); }
    }, 2500);
    return () => clearInterval(timer);
  }, [job?.id, job?.status]);

  async function confirmIdentity() {
    if (!job || selectedTrack === null) return;
    try {
      await api(`/api/jobs/${job.id}/tracks/${selectedTrack}/label`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_name: playerName, role: playerName.toLowerCase().includes("manager") ? "manager" : "player", team_name: teamName }),
      });
      setTracks(await api<{ tracks: Track[] }>(`/api/jobs/${job.id}/tracks`).then(x => x.tracks));
      setMessage(`Track ${selectedTrack} confirmed as ${playerName}.`);
    } catch (e) { setMessage(e instanceof Error ? e.message : "Could not label track."); }
  }

  async function createClip() {
    if (!job) return;
    try {
      const result = await api<{ clip_id: string }>(`/api/jobs/${job.id}/clips`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_seconds: Number(start), end_seconds: Number(end), title: `${playerName} highlight`, category: "player_highlight", aspect_ratio: "9:16", selected_track_id: selectedTrack }),
      });
      setMessage(`Rendering clip ${result.clip_id}. It will appear in the list when refreshed.`);
      window.setTimeout(async () => {
        try { setClips(await api<{ clips: Clip[] }>(`/api/jobs/${job.id}/clips`).then(x => x.clips)); } catch { /* status can still be pending */ }
      }, 4000);
    } catch (e) { setMessage(e instanceof Error ? e.message : "Clip render failed."); }
  }

  async function refreshClips() {
    if (!job) return;
    try { setClips(await api<{ clips: Clip[] }>(`/api/jobs/${job.id}/clips`).then(x => x.clips)); }
    catch (e) { setMessage(e instanceof Error ? e.message : "Could not refresh clips."); }
  }

  return (
    <main className="mx-auto max-w-4xl space-y-6 rounded-2xl border p-6">
      <header><p className="text-sm font-semibold uppercase tracking-widest">PitchPerfect AI</p><h1 className="text-3xl font-bold">Football highlight studio</h1><p>Upload a match, confirm a player track, and render vertical clips.</p></header>
      <section className="space-y-3 rounded-xl border p-4">
        <h2 className="text-xl font-semibold">1. Upload full match</h2>
        <label className="block text-sm">Match name<input className="mt-1 block w-full rounded border p-2" value={matchName} onChange={e => setMatchName(e.target.value)} /></label>
        <input type="file" accept="video/*,.mkv" onChange={e => setFile(e.target.files?.[0] ?? null)} />
        <button disabled={busy || !file} onClick={uploadMatch} className="rounded-lg bg-black px-4 py-2 text-white disabled:opacity-40">Upload and analyse</button>
        {job && <div><p>{job.status} · {job.stage} · {job.progress}%</p><progress className="w-full" max={100} value={job.progress} /></div>}
      </section>
      {job?.status === "completed" && <>
        <section className="space-y-3 rounded-xl border p-4">
          <h2 className="text-xl font-semibold">2. Confirm player identity</h2>
          <p className="text-sm">Select a visual track. Track IDs can change across broadcast cuts, so confirm carefully.</p>
          <select className="w-full rounded border p-2" value={selectedTrack ?? ""} onChange={e => setSelectedTrack(e.target.value ? Number(e.target.value) : null)}>
            <option value="">Choose a track</option>{tracks.map(t => <option key={t.track_id} value={t.track_id}>Track {t.track_id} · {t.observations} observations · {Math.round(t.first_seen_seconds)}s–{Math.round(t.last_seen_seconds)}s {t.player_name ? `· ${t.player_name}` : ""}</option>)}
          </select>
          <div className="grid gap-3 sm:grid-cols-2"><input className="rounded border p-2" value={playerName} onChange={e => setPlayerName(e.target.value)} placeholder="Player or manager name" /><input className="rounded border p-2" value={teamName} onChange={e => setTeamName(e.target.value)} placeholder="Team" /></div>
          <button disabled={selectedTrack === null} onClick={confirmIdentity} className="rounded-lg border px-4 py-2 disabled:opacity-40">Confirm identity</button>
        </section>
        <section className="space-y-3 rounded-xl border p-4">
          <h2 className="text-xl font-semibold">3. Create player-aware vertical clip</h2>
          <div className="grid gap-3 sm:grid-cols-2"><label>Start seconds<input type="number" min="0" className="mt-1 block w-full rounded border p-2" value={start} onChange={e => setStart(e.target.value)} /></label><label>End seconds<input type="number" min="0.5" className="mt-1 block w-full rounded border p-2" value={end} onChange={e => setEnd(e.target.value)} /></label></div>
          <button onClick={createClip} className="rounded-lg bg-black px-4 py-2 text-white">Render 9:16 clip</button>
        </section>
        <section className="space-y-3 rounded-xl border p-4"><div className="flex items-center justify-between"><h2 className="text-xl font-semibold">4. Clip library</h2><button onClick={refreshClips} className="rounded border px-3 py-1">Refresh</button></div>
          {clips.map(c => <article key={c.id} className="flex flex-wrap items-center justify-between gap-2 border-t py-3"><div><p className="font-medium">{c.title}</p><p className="text-sm">{c.category} · {c.start_seconds.toFixed(1)}s–{c.end_seconds.toFixed(1)}s · {c.status}</p></div>{c.status === "ready" && <a className="underline" href={`${API}/api/clips/${c.id}/download`}>Download MP4</a>}</article>)}
        </section>
      </>}
      {message && <p role="status" className="rounded-lg bg-gray-100 p-3 text-sm text-gray-900">{message}</p>}
      <p className="text-xs text-gray-500">This frontend is a starter component. Goal, celebration, teammate and manager moments still require human review or a validated football-specific model.</p>
    </main>
  );
}
