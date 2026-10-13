"""
agents/reel_producer.py
────────────────────────
Agent 8 — ReelProducerAgent
FACE — Facebook Autonomous Content Engine

Responsibilities:
  1. For each reel in content_plan.json, call Gemini to generate a shot list /
     visual direction plan (reel_script_visual_director.txt)
  2. Gather per-segment assets: stock video (Pixabay if configured), generated images, or
     text cards
  3. Generate TTS voiceover per segment, concatenate, and produce subtitle
     timings
  4. Assemble the reel: clips → colour grade → animate → voiceover + music →
     burn subtitles → export 1080×1920 H.264/AAC ≤60 s
  5. Quality-check the exported MP4 (duration, resolution, size)
  6. Extract thumbnail frame, apply hook text overlay
  7. Write state/reel_plans.json and patch state/generated_content.json with
     reel_path, reel_thumbnail_path, reel_ready fields
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from utils.asset_cache import get_asset_cache
from utils.audio_utils import (
    generate_subtitle_timings,
    generate_tts,
    get_audio_duration,
    get_music_track,
)
from utils.config_loader import build_prompt_context, get_config
from utils.gemini_client import GeminiQuotaExhaustedError, get_gemini_client
from agents.visual_director import (
    HF_FLUX_MODEL, HF_SDXL_MODEL, _provider_hf, _provider_pollinations, _provider_pixabay,
)
from utils.image_utils import add_text_overlay
from utils.logger import get_logger, log_exception
from utils.rate_limiter import GEMINI_LIMITER
from utils.state_manager import (
    read_state,
    state_exists,
    write_state,
)
from utils.video_utils import (
    apply_animation,
    apply_color_grade_video,
    assemble_reel,
    burn_subtitles,
    create_text_card,
    export_reel,
    mix_audio,
    verify_reel,
)

log = get_logger("reel_producer")

# ── Directory constants ────────────────────────────────────────────────────────
REEL_FINAL = Path("assets/videos/final")
REEL_TEMP = Path("assets/videos/temp")
AUDIO_TEMP = Path("assets/audio/temp")
IMAGES_REEL = Path("assets/images/reel_frames")
THUMB_FINAL = Path("assets/images/final")

for _d in (REEL_FINAL, REEL_TEMP, AUDIO_TEMP, IMAGES_REEL, THUMB_FINAL):
    _d.mkdir(parents=True, exist_ok=True)

# ── Inline image generation for reel segments ─────────────────────────────────
# Reuse the same authenticated Pollinations/Hugging Face providers and credit checks
# as VisualDirector, avoiding stale endpoints and repeated exhausted-credit retries.


def _fit_image_for_reel(source_path: Path, output_path: Path) -> Path:
    """Convert and crop any supported source image to the reel's 9:16 canvas."""
    from PIL import Image, ImageOps

    with Image.open(source_path) as source:
        resampling = getattr(Image, "Resampling", Image)
        frame = ImageOps.fit(
            source.convert("RGB"), (1080, 1920), method=resampling.LANCZOS
        )
        frame.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path


def _generate_segment_image(
    prompt: str, reel_id: str, seg_idx: int, config: dict | None = None
) -> Path | None:
    """Generate one vertical reel frame with the current provider waterfall.

    Pexels is disabled. Pixabay is used only when its API key is configured.
    """
    cfg = config or get_config()
    output_path = IMAGES_REEL / f"{reel_id}_seg{seg_idx}.jpg"
    provider_post_id = f"{reel_id}_seg{seg_idx}"
    pollinations_model = (
        os.environ.get("POLLINATIONS_IMAGE_MODEL", "").strip()
        or str(cfg.get("api", {}).get("pollinations_image_model", "flux")).strip()
        or "flux"
    )

    raw_path = _provider_pollinations(
        prompt, 1080, 1920, provider_post_id, 1, model=pollinations_model
    )
    if raw_path is None:
        for attempt, (model_id, tag) in enumerate(
            ((HF_FLUX_MODEL, "Seg-P2-FLUX"), (HF_SDXL_MODEL, "Seg-P3-SDXL")),
            start=2,
        ):
            raw_path = _provider_hf(
                model_id, prompt, 1080, 1920, provider_post_id, attempt, tag
            )
            if raw_path is not None:
                break

    if raw_path is None and os.environ.get("PIXABAY_API_KEY", "").strip():
        raw_path = _provider_pixabay(prompt[:100], "portrait", provider_post_id, 4)

    if raw_path is not None:
        try:
            _fit_image_for_reel(Path(raw_path), output_path)
            log.info("Reel segment image ready via shared provider: %s", output_path)
            return output_path
        except Exception as exc:
            log.warning("Could not prepare generated reel frame %s: %s", provider_post_id, exc)

    log.error(
        "All configured non-Pexels image providers failed for segment %d of reel %s",
        seg_idx, reel_id,
    )
    return None


# ── Shot plan generation ──────────────────────────────────────────────────────


def _generate_shot_plan(
    gemini,
    reel_id: str,
    reel_script: dict,
    visual_brief: dict | None,
    context: dict,
    config: dict,
) -> dict:
    """
    Call Gemini with reel_script_visual_director.txt to produce a shot list.
    Falls back to a minimal 1-segment stub on error.
    """
    GEMINI_LIMITER.acquire()
    runtime_ctx = {
        **context,
        "reel_id": reel_id,
        "reel_script_json": json.dumps(reel_script, ensure_ascii=False),
        "visual_brief_json": json.dumps(visual_brief or {}, ensure_ascii=False),
        "audio_mood": reel_script.get("audio_mood", "energetic"),
        "estimated_duration": reel_script.get("estimated_duration", 30),
    }
    try:
        plan = gemini.generate_json("reel_script_visual_director.txt", **runtime_ctx)
        plan["reel_id"] = reel_id
        return plan
    except GeminiQuotaExhaustedError:
        raise
    except Exception as exc:
        log_exception(log, f"Shot plan generation failed for reel {reel_id}", exc)
        # Minimal fallback shot plan
        hook = reel_script.get("hook", "")
        body = reel_script.get("body_segments", [])
        cta = reel_script.get("cta", "")
        segments = []
        all_texts = [hook] + [
            (s["text"] if isinstance(s, dict) else str(s)) for s in body
        ] + [cta]
        for i, text in enumerate(all_texts):
            segments.append({
                "segment_index": i,
                "visual_type": "text_card",
                "duration": 5,
                "voiceover_text": text,
                "tts_voice": config.get("tts_voice", "en-US-GuyNeural"),
                "tts_speed": "+0%",
                "animation": "fade_in",
                "color_grade": "neutral",
                "music_mood": reel_script.get("audio_mood", "energetic"),
                "text_card_content": text,
                "text_card_style": "bold_white",
            })
        return {
            "reel_id": reel_id,
            "segments": segments,
            "thumbnail_segment_index": 0,
            "hook_text_for_thumbnail": hook,
        }


# ── Asset gathering ───────────────────────────────────────────────────────────


def _gather_segment_asset(
    segment: dict,
    reel_id: str,
    seg_idx: int,
    config: dict,
) -> str | None:
    """
    Gather or note the asset for a single segment.
    Returns a file path string, or None for text_card (generated at assembly).
    """
    visual_type = segment.get("visual_type", "text_card")

    if visual_type == "stock_video":
        query = segment.get("stock_video_search_query", config.get("niche", ""))

        # Pixabay is optional; if no stock video is available, generate a frame instead.
        if os.environ.get("PIXABAY_API_KEY", "").strip():
            try:
                cache = get_asset_cache()
                clip_path = cache.download_pixabay_video(
                    query=query,
                    orientation="portrait",
                )
                log.info(
                    "Pixabay stock video downloaded: %s",
                    clip_path,
                )
                segment["_asset_path"] = str(clip_path)
                return str(clip_path)
            except Exception as exc:
                log_exception(
                    log,
                    f"Pixabay stock video failed for segment {seg_idx}",
                    exc,
                )

        # Fallback: generate image instead
        log.info("Falling back to generated image for seg %d", seg_idx)
        prompt = segment.get("stock_video_search_query", config.get("niche", "scenic"))
        path = _generate_segment_image(prompt, reel_id, seg_idx, config=config)
        if path:
            segment["visual_type"] = "generated_image"
            segment["_asset_path"] = str(path)
        return str(path) if path else None

    elif visual_type == "generated_image":
        prompt = segment.get("image_prompt", segment.get("voiceover_text", ""))
        path = _generate_segment_image(prompt, reel_id, seg_idx, config=config)
        if path:
            segment["_asset_path"] = str(path)
            return str(path)
        return None

    elif visual_type == "text_card":
        # No asset to download — create_text_card() handles this at assembly
        return None

    log.warning("Unknown visual_type '%s' for seg %d — treating as text_card", visual_type, seg_idx)
    return None


# ── Voiceover generation ──────────────────────────────────────────────────────


def _build_voiceover(
    segments: list[dict],
    reel_id: str,
    config: dict,
) -> tuple[str | None, list[dict]]:
    """
    Generate TTS for all segments that have voiceover_text.
    Concatenates individual MP3s → combined voiceover file.
    Returns (combined_vo_path, updated_segments_with_audio_paths).
    """
    seg_audio_paths: list[str] = []

    for i, seg in enumerate(segments):
        text = seg.get("voiceover_text", "").strip()
        if not text:
            seg["_audio_path"] = None
            continue

        voice = seg.get("tts_voice", config.get("tts_voice", "en-US-GuyNeural"))
        speed = seg.get("tts_speed", "+0%")
        out_path = str(AUDIO_TEMP / f"{reel_id}_seg{i}.mp3")

        try:
            audio_path = generate_tts(
                text=text,
                voice=voice,
                speed=speed,
                output_path=out_path,
            )
            seg["_audio_path"] = audio_path
            seg_audio_paths.append(audio_path)
            dur = get_audio_duration(audio_path)
            log.info(
                "TTS seg %d — %.1f s — voice=%s — path=%s", i, dur, voice, audio_path
            )
        except Exception as exc:
            log_exception(log, f"TTS failed for seg {i} reel {reel_id}", exc)
            seg["_audio_path"] = None

    if not seg_audio_paths:
        log.warning("No TTS audio generated for reel %s", reel_id)
        return None, segments

    # Concatenate segment audio files → combined voiceover
    combined_vo_path = str(AUDIO_TEMP / f"{reel_id}_vo.mp3")
    try:
        # Use pydub for simple concatenation (available via requirements.txt)
        from pydub import AudioSegment  # type: ignore

        combined = AudioSegment.empty()
        for ap in seg_audio_paths:
            combined += AudioSegment.from_mp3(ap)
        combined.export(combined_vo_path, format="mp3")
        log.info(
            "Combined voiceover: %s (%.1f s)",
            combined_vo_path,
            get_audio_duration(combined_vo_path),
        )
    except Exception as exc:
        log_exception(log, f"Voiceover concatenation failed for {reel_id}", exc)
        # If concatenation fails, use the first segment audio as fallback
        if seg_audio_paths:
            shutil.copy2(seg_audio_paths[0], combined_vo_path)
            log.warning("Using first segment audio as voiceover fallback")
        else:
            return None, segments

    return combined_vo_path, segments


# ── Clip assembly ─────────────────────────────────────────────────────────────


def _build_clips(
    segments: list[dict],
    reel_id: str,
) -> list[Any]:
    """
    Build a list of moviepy clips from the shot plan segments.
    Returns list of processed VideoClip objects.
    """
    try:
        from moviepy.editor import ImageClip, VideoFileClip  # type: ignore
    except ImportError:
        log.critical("moviepy not available — cannot build clips")
        return []

    clips: list[Any] = []

    for i, seg in enumerate(segments):
        visual_type = seg.get("visual_type", "text_card")
        duration = float(seg.get("duration", 5))
        animation = seg.get("animation", "fade_in")
        color_grade = seg.get("color_grade", "neutral")

        try:
            if visual_type == "stock_video" and seg.get("_asset_path"):
                raw = VideoFileClip(seg["_asset_path"])
                # Trim to segment duration; loop if clip is shorter
                if raw.duration < duration:
                    raw = raw.loop(duration=duration)
                clip = raw.subclip(0, min(duration, raw.duration))

            elif visual_type == "generated_image" and seg.get("_asset_path"):
                clip = ImageClip(seg["_asset_path"], duration=duration)

            elif visual_type == "text_card":
                text_content = seg.get("text_card_content", seg.get("voiceover_text", ""))
                style = seg.get("text_card_style", "bold_white")
                clip = create_text_card(text=text_content, style=style, duration=duration)

            else:
                # Fallback: blank text card with voiceover text
                fallback_text = seg.get("voiceover_text", "")
                log.warning(
                    "Segment %d has no asset — using text card fallback (reel %s)",
                    i, reel_id,
                )
                clip = create_text_card(
                    text=fallback_text, style="bold_white", duration=duration
                )

            # Apply animation and colour grade
            clip = apply_animation(clip, animation)
            clip = apply_color_grade_video(clip, color_grade)
            clips.append(clip)
            log.debug("Built clip %d: visual_type=%s duration=%.1f", i, visual_type, duration)

        except Exception as exc:
            log_exception(log, f"Failed to build clip {i} for reel {reel_id}", exc)
            # Insert a blank text card so the reel still assembles
            fallback_text = seg.get("voiceover_text", f"[Slide {i + 1}]")
            try:
                clip = create_text_card(text=fallback_text, style="bold_white", duration=duration)
                clips.append(clip)
                log.warning("Used fallback text card for segment %d", i)
            except Exception as inner_exc:
                log_exception(log, f"Even fallback clip failed for seg {i}", inner_exc)

    return clips


# ── Thumbnail extraction ──────────────────────────────────────────────────────


def _generate_thumbnail(
    reel_path: str,
    reel_id: str,
    shot_plan: dict,
    config: dict,
) -> str | None:
    """
    Extract a frame from the assembled reel at the midpoint of the
    thumbnail segment, apply hook text overlay, save to
    assets/images/final/{reel_id}_thumb.jpg.
    """
    thumb_seg_idx: int = shot_plan.get("thumbnail_segment_index", 0)
    hook_text: str = shot_plan.get("hook_text_for_thumbnail", "")
    thumb_path = str(THUMB_FINAL / f"{reel_id}_thumb.jpg")

    try:
        from moviepy.editor import VideoFileClip  # type: ignore

        vc = VideoFileClip(reel_path)
        segments = shot_plan.get("segments", [])

        # Calculate the midpoint timestamp of the thumbnail segment
        elapsed = 0.0
        for i, seg in enumerate(segments):
            seg_dur = float(seg.get("duration", 5))
            if i == thumb_seg_idx:
                t = elapsed + seg_dur / 2.0
                break
            elapsed += seg_dur
        else:
            t = vc.duration / 2.0

        t = min(t, vc.duration - 0.1)
        frame = vc.get_frame(t)
        vc.close()

        # Save raw frame as JPEG
        from PIL import Image  # type: ignore
        import numpy as np

        raw_thumb_path = str(IMAGES_REEL / f"{reel_id}_thumb_raw.jpg")
        Image.fromarray(np.uint8(frame)).save(raw_thumb_path, "JPEG", quality=95)

        # Apply hook text overlay
        if hook_text:
            result = add_text_overlay(
                image_path=raw_thumb_path,
                text=hook_text,
                position="bottom",
                font_size=84,
                color="auto",
                shadow=True,
                backdrop=True,
            )
            shutil.copy2(result, thumb_path)
        else:
            shutil.copy2(raw_thumb_path, thumb_path)

        log.info("Thumbnail saved: %s", thumb_path)
        return thumb_path

    except Exception as exc:
        log_exception(log, f"Thumbnail generation failed for reel {reel_id}", exc)
        return None


# ── Temp file cleanup ─────────────────────────────────────────────────────────


def _cleanup_temp_files(reel_id: str) -> None:
    """Remove all temp assets created during reel production for reel_id."""
    patterns = [
        AUDIO_TEMP / f"{reel_id}_*.mp3",
        IMAGES_REEL / f"{reel_id}_*.jpg",
        REEL_TEMP / f"{reel_id}*",
    ]
    from glob import glob

    for pattern in patterns:
        for f in glob(str(pattern)):
            try:
                os.remove(f)
                log.debug("Cleaned up temp file: %s", f)
            except OSError as exc:
                log.warning("Could not remove temp file %s: %s", f, exc)


# ── Main agent ────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("=== ReelProducerAgent starting ===")
    config = get_config()
    context = build_prompt_context()
    gemini = get_gemini_client()

    # ── Load inputs ──────────────────────────────────────────────────────────
    if not state_exists("content_plan.json"):
        log.critical("state/content_plan.json missing — cannot run ReelProducerAgent")
        return
    if not state_exists("generated_content.json"):
        log.critical("state/generated_content.json missing — cannot run ReelProducerAgent")
        return

    content_plan = read_state("content_plan.json")
    generated_content_data = read_state("generated_content.json")
    content_by_id: dict[str, dict] = {
        item["post_id"]: item
        for item in generated_content_data.get("content", [])
    }

    # Load visual briefs (may not exist if VisualDirectorAgent hasn't run yet)
    if state_exists("visual_briefs.json"):
        visual_briefs_data = read_state("visual_briefs.json")
        visual_briefs: dict[str, dict] = visual_briefs_data.get("briefs", {})
    else:
        visual_briefs = {}
        log.warning("state/visual_briefs.json not found — proceeding without visual briefs")

    # Identify reels
    reels: list[dict] = content_plan.get("reels", [])
    if not reels:
        # Fallback: filter from posts
        reels = [p for p in content_plan.get("posts", []) if p.get("format") == "reel"]

    reels_with_content = [
        r for r in reels if r["post_id"] in content_by_id
        and content_by_id[r["post_id"]].get("reel_script") is not None
    ]

    log.info("Reels to produce: %d / %d total", len(reels_with_content), len(reels))

    all_shot_plans: dict[str, dict] = {}
    reel_updates: dict[str, dict] = {}

    for reel_post in reels_with_content:
        reel_id = reel_post["post_id"]
        content = content_by_id[reel_id]
        reel_script = content.get("reel_script", {})
        visual_brief = visual_briefs.get(reel_id)

        log.info("--- Processing reel %s ---", reel_id)
        t_start = time.time()

        # ── Step 1: Generate shot plan ────────────────────────────────────
        log.info("[%s] Step 1: Generating shot plan", reel_id)
        try:
            shot_plan = _generate_shot_plan(
                gemini, reel_id, reel_script, visual_brief, context, config
            )
        except GeminiQuotaExhaustedError:
            log.critical("Gemini quota exhausted — halting ReelProducerAgent")
            break
        except Exception as exc:
            log_exception(log, f"Shot plan failed for {reel_id}", exc)
            reel_updates[reel_id] = {
                "reel_path": None,
                "reel_thumbnail_path": None,
                "reel_ready": False,
            }
            continue

        all_shot_plans[reel_id] = shot_plan
        segments: list[dict] = shot_plan.get("segments", [])
        log.info("[%s] Shot plan: %d segments", reel_id, len(segments))

        # ── Step 2: Asset gathering ───────────────────────────────────────
        log.info("[%s] Step 2: Gathering segment assets", reel_id)
        for i, seg in enumerate(segments):
            asset_path = _gather_segment_asset(seg, reel_id, i, config)
            if asset_path:
                seg["_asset_path"] = asset_path

        # ── Step 3: Voiceover generation ──────────────────────────────────
        log.info("[%s] Step 3: Generating voiceover", reel_id)
        combined_vo_path, segments = _build_voiceover(segments, reel_id, config)
        shot_plan["segments"] = segments  # Update with audio path annotations

        # Generate subtitle timings
        subtitle_timings: list[dict] = []
        if combined_vo_path:
            try:
                subtitle_timings = generate_subtitle_timings(segments, combined_vo_path)
                log.info("[%s] Subtitle timings: %d entries", reel_id, len(subtitle_timings))
            except Exception as exc:
                log_exception(log, f"Subtitle timing generation failed for {reel_id}", exc)

        # ── Step 4: Reel assembly ─────────────────────────────────────────
        log.info("[%s] Step 4: Assembling reel clips", reel_id)
        clips = _build_clips(segments, reel_id)

        if not clips:
            log.error("[%s] No clips built — cannot assemble reel", reel_id)
            reel_updates[reel_id] = {
                "reel_path": None,
                "reel_thumbnail_path": None,
                "reel_ready": False,
            }
            continue

        output_path = str(REEL_FINAL / f"{reel_id}.mp4")

        try:
            # Assemble clips
            log.info("[%s] Concatenating %d clips...", reel_id, len(clips))
            t_assemble = time.time()
            final_clip = assemble_reel(clips)
            log.info(
                "[%s] Clips assembled in %.1f s", reel_id, time.time() - t_assemble
            )

            # Music
            music_mood = segments[0].get("music_mood", reel_script.get("audio_mood", "energetic"))
            music_path = get_music_track(mood=music_mood, cache_dir="assets/audio")
            log.info("[%s] Music: %s (mood=%s)", reel_id, music_path, music_mood)

            # Mix voiceover + music; prefer voiceover volume
            if combined_vo_path and music_path:
                final_audio = mix_audio(combined_vo_path, music_path)
                final_clip = final_clip.set_audio(final_audio)
            elif combined_vo_path:
                try:
                    from moviepy.editor import AudioFileClip  # type: ignore
                    vo_audio = AudioFileClip(combined_vo_path)
                    final_clip = final_clip.set_audio(vo_audio)
                except Exception as exc:
                    log_exception(log, f"Setting voiceover audio failed for {reel_id}", exc)

            # Burn subtitles
            if subtitle_timings:
                try:
                    final_clip = burn_subtitles(final_clip, subtitle_timings)
                    log.info("[%s] Subtitles burned", reel_id)
                except Exception as exc:
                    log_exception(log, f"Subtitle burn failed for {reel_id}", exc)

            # Export
            log.info("[%s] Exporting reel to %s...", reel_id, output_path)
            t_export = time.time()
            export_reel(final_clip, output_path)
            log.info(
                "[%s] Export complete in %.1f s", reel_id, time.time() - t_export
            )

        except Exception as exc:
            log_exception(log, f"Reel assembly/export failed for {reel_id}", exc)
            reel_updates[reel_id] = {
                "reel_path": None,
                "reel_thumbnail_path": None,
                "reel_ready": False,
            }
            continue

        # ── Step 5: Quality check ─────────────────────────────────────────
        log.info("[%s] Step 5: Verifying reel", reel_id)
        result = verify_reel(output_path)
        duration_ok = result.get("duration", 999) <= 62
        resolution_ok = tuple(result.get("resolution", (0, 0))) == (1080, 1920)
        size_ok = result.get("size_mb", 999) < 100

        if not duration_ok:
            log.error(
                "[%s] Duration exceeds limit: %.1f s", reel_id, result.get("duration")
            )
        if not resolution_ok:
            log.error(
                "[%s] Resolution mismatch: %s", reel_id, result.get("resolution")
            )
        if not size_ok:
            log.error(
                "[%s] File too large: %.1f MB", reel_id, result.get("size_mb")
            )

        reel_valid = duration_ok and resolution_ok and size_ok

        if not reel_valid:
            log.error("[%s] Reel failed quality check — marking as failed", reel_id)
            reel_updates[reel_id] = {
                "reel_path": output_path,
                "reel_thumbnail_path": None,
                "reel_ready": False,
                "reel_qa": result,
            }
            _cleanup_temp_files(reel_id)
            continue

        log.info(
            "[%s] Reel verified ✓ (%.1f s, %dx%d, %.1f MB)",
            reel_id,
            result.get("duration"),
            *result.get("resolution", (0, 0)),
            result.get("size_mb"),
        )

        # ── Step 6: Generate thumbnail ────────────────────────────────────
        log.info("[%s] Step 6: Generating thumbnail", reel_id)
        thumb_path = _generate_thumbnail(output_path, reel_id, shot_plan, config)

        total_time = time.time() - t_start
        log.info("[%s] Reel production complete in %.1f s", reel_id, total_time)

        reel_updates[reel_id] = {
            "reel_path": output_path,
            "reel_thumbnail_path": thumb_path,
            "reel_ready": True,
            "reel_qa": result,
        }

        # Cleanup temp assets
        _cleanup_temp_files(reel_id)

    # ── Write state/reel_plans.json ──────────────────────────────────────────
    write_state(
        "reel_plans.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date": content_plan.get("date", ""),
            "plans": all_shot_plans,
        },
    )
    log.info("state/reel_plans.json written (%d plans)", len(all_shot_plans))

    # ── Patch state/generated_content.json ───────────────────────────────────
    current_content = read_state("generated_content.json")
    for item in current_content.get("content", []):
        pid = item["post_id"]
        if pid in reel_updates:
            item.update(reel_updates[pid])
    write_state("generated_content.json", current_content)
    log.info(
        "state/generated_content.json patched for %d reels", len(reel_updates)
    )

    successful = sum(1 for v in reel_updates.values() if v.get("reel_ready"))
    log.info(
        "=== ReelProducerAgent complete — %d/%d reels ready ===",
        successful, len(reel_updates),
    )


if __name__ == "__main__":
    main()
