"""Persistence.

The behaviour worth guarding here is identity: a segment is unique on
``(session_id, track, seq)``, which is what makes the draft-to-final transition an
update of one line rather than two lines in the transcript. Everything else -- the
crash-recovery story, the panel's replay, the retroactive rename -- rests on that.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from meetscribe.core.models import (
    ActionItem,
    ConsentMode,
    SegmentState,
    SessionState,
    Speaker,
    Summary,
    TrackKind,
    TranscriptSegment,
    new_id,
)
from meetscribe.errors import SessionNotFoundError, StoreError
from meetscribe.store.sqlite_store import SCHEMA_VERSION, SqliteStore


def segment(session_id: str, seq: int, text: str, **kw) -> TranscriptSegment:
    defaults = dict(
        segment_id=new_id(),
        session_id=session_id,
        track=TrackKind.LOOPBACK,
        start_s=float(seq),
        end_s=float(seq) + 1.0,
        speaker_id="p1",
        speaker_name="Participante 1",
        seq=seq,
        state=SegmentState.FINAL,
    )
    defaults.update(kw)
    return TranscriptSegment(text=text, **defaults)


@pytest.fixture
def saved(store, session):
    store.create_session(session)
    return session


# ------------------------------------------------------------------- sessions


def test_create_and_read_back_every_field(store, session):
    store.create_session(session)
    loaded = store.get_session(session.session_id)
    assert loaded == session


def test_update_session_requires_it_to_exist(store, session):
    with pytest.raises(SessionNotFoundError):
        store.update_session(session)


def test_update_session_persists_the_end_state(store, saved):
    ended = dataclasses.replace(
        saved,
        state=SessionState.STOPPED,
        ended_at=dt.datetime(2026, 3, 1, 11, 0, tzinfo=dt.UTC),
        consent=ConsentMode.ALL_PARTIES,
    )
    store.update_session(ended)
    loaded = store.get_session(saved.session_id)
    assert loaded.state is SessionState.STOPPED
    assert loaded.duration_s == pytest.approx(3600)
    assert loaded.consent is ConsentMode.ALL_PARTIES


def test_resolve_session_accepts_a_prefix(store, saved):
    assert store.resolve_session(saved.session_id[:6]).session_id == saved.session_id
    assert store.resolve_session(saved.name).session_id == saved.session_id


def test_resolve_session_refuses_to_guess_between_matches(store, session):
    a = dataclasses.replace(session, session_id="abc111", name="Uno")
    b = dataclasses.replace(session, session_id="abc222", name="Dos")
    store.create_session(a)
    store.create_session(b)
    with pytest.raises(SessionNotFoundError, match="ambiguous"):
        store.resolve_session("abc")


def test_resolve_session_reports_an_unknown_id_clearly(store):
    with pytest.raises(SessionNotFoundError) as excinfo:
        store.resolve_session("nope")
    assert "nope" in excinfo.value.user_message


def test_sessions_are_listed_newest_first(store, session):
    for i in range(3):
        store.create_session(
            dataclasses.replace(
                session,
                session_id=f"s{i}",
                started_at=session.started_at + dt.timedelta(hours=i),
            )
        )
    assert [s.session_id for s in store.list_sessions()] == ["s2", "s1", "s0"]


def test_deleting_a_session_cascades_to_its_content(store, saved):
    store.add_segment(segment(saved.session_id, 1, "hola"))
    store.upsert_speaker(saved.session_id, Speaker("p1", "P1", TrackKind.LOOPBACK))
    store.delete_session(saved.session_id)
    assert store.get_segments(saved.session_id) == []
    assert store.get_speakers(saved.session_id) == []


# ------------------------------------------------------------------- segments


def test_a_final_segment_replaces_its_draft_in_place(store, saved):
    draft = segment(saved.session_id, 5, "hola qu", state=SegmentState.DRAFT)
    store.add_segment(draft)
    final = segment(saved.session_id, 5, "hola qué tal", state=SegmentState.FINAL)
    store.replace_segment(final)

    rows = store.get_segments(saved.session_id)
    assert len(rows) == 1
    assert rows[0].text == "hola qué tal"
    assert rows[0].state is SegmentState.FINAL
    # The id survives, so anything already holding a reference still resolves.
    assert rows[0].segment_id == draft.segment_id


def test_a_late_draft_can_never_overwrite_confirmed_text(store, saved):
    store.replace_segment(segment(saved.session_id, 5, "texto confirmado"))
    store.replace_segment(
        segment(saved.session_id, 5, "texto provisional", state=SegmentState.DRAFT)
    )
    rows = store.get_segments(saved.session_id)
    assert rows[0].text == "texto confirmado"
    assert rows[0].state is SegmentState.FINAL


def test_segments_from_both_tracks_interleave_by_time(store, saved):
    store.add_segment(segment(saved.session_id, 1, "remoto", start_s=5.0, end_s=6.0))
    store.add_segment(
        segment(
            saved.session_id, 1, "yo", track=TrackKind.MIC, start_s=2.0, end_s=3.0,
            speaker_id="yo", speaker_name="Yo",
        )
    )
    assert [s.text for s in store.get_segments(saved.session_id)] == ["yo", "remoto"]


def test_final_only_filter(store, saved):
    store.add_segment(segment(saved.session_id, 1, "a"))
    store.add_segment(segment(saved.session_id, 2, "b", state=SegmentState.DRAFT))
    assert len(store.get_segments(saved.session_id)) == 2
    assert len(store.get_segments(saved.session_id, final_only=True)) == 1


def test_drop_drafts_keeps_the_archive_honest(store, saved):
    store.add_segment(segment(saved.session_id, 1, "confirmado"))
    store.add_segment(segment(saved.session_id, 2, "provisional", state=SegmentState.DRAFT))
    assert store.drop_drafts(saved.session_id) == 1
    assert [s.text for s in store.get_segments(saved.session_id)] == ["confirmado"]


# ------------------------------------------------------------------- speakers


def test_rename_applies_retroactively_to_every_line(store, saved):
    store.upsert_speaker(saved.session_id, Speaker("p1", "Participante 1", TrackKind.LOOPBACK))
    for i in range(3):
        store.add_segment(segment(saved.session_id, i, f"linea {i}"))

    touched = store.rename_speaker(saved.session_id, "p1", "María")
    assert touched == 3
    assert all(s.speaker_name == "María" for s in store.get_segments(saved.session_id))
    assert store.get_speakers(saved.session_id)[0].display_name == "María"


def test_rename_rejects_a_blank_name(store, saved):
    store.upsert_speaker(saved.session_id, Speaker("p1", "P1", TrackKind.LOOPBACK))
    with pytest.raises(StoreError):
        store.rename_speaker(saved.session_id, "p1", "   ")


def test_rename_of_an_unknown_speaker_is_an_error(store, saved):
    with pytest.raises(StoreError, match="no speaker"):
        store.rename_speaker(saved.session_id, "p9", "Ana")


def test_reassign_applies_the_refinement_and_carries_the_display_name(store, saved):
    store.upsert_speaker(saved.session_id, Speaker("p1", "Ana", TrackKind.LOOPBACK))
    store.upsert_speaker(saved.session_id, Speaker("p2", "Luis", TrackKind.LOOPBACK))
    store.add_segment(segment(saved.session_id, 1, "uno", speaker_id="p2", speaker_name="Luis"))
    store.add_segment(segment(saved.session_id, 2, "dos", speaker_id="p2", speaker_name="Luis"))

    changed = store.reassign_speakers(saved.session_id, {("loopback", 1): "p1"})
    assert changed == 1
    rows = store.get_segments(saved.session_id)
    assert (rows[0].speaker_id, rows[0].speaker_name) == ("p1", "Ana")
    assert rows[1].speaker_id == "p2"


def test_reassign_with_nothing_to_do(store, saved):
    assert store.reassign_speakers(saved.session_id, {}) == 0


def test_upsert_speaker_updates_the_count(store, saved):
    store.upsert_speaker(saved.session_id, Speaker("p1", "P1", TrackKind.LOOPBACK))
    store.upsert_speaker(
        saved.session_id, Speaker("p1", "P1", TrackKind.LOOPBACK, utterance_count=7)
    )
    speakers = store.get_speakers(saved.session_id)
    assert len(speakers) == 1
    assert speakers[0].utterance_count == 7


# ----------------------------------------------- summaries and action items


def summary_for(session_id: str, **kw) -> Summary:
    defaults = dict(
        provider="fake",
        model="fake-1",
        generated_at=dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.UTC),
        executive="Se revisó el avance.",
        topics=("facturación",),
        decisions=("escalar el bloqueo",),
        action_items=(ActionItem(text="hablar con el banco", owner="Yo", due="el viernes"),),
        open_questions=("¿quién firma?",),
        follow_ups=("reunión el lunes",),
    )
    defaults.update(kw)
    return Summary(session_id=session_id, **defaults)


def test_summary_round_trip(store, saved):
    original = summary_for(saved.session_id)
    store.save_summary(original)
    loaded = store.get_summary(saved.session_id)
    assert loaded == original


def test_saving_a_summary_twice_replaces_it(store, saved):
    store.save_summary(summary_for(saved.session_id))
    store.save_summary(summary_for(saved.session_id, executive="Otra cosa."))
    assert store.get_summary(saved.session_id).executive == "Otra cosa."


def test_saving_a_summary_records_its_action_items(store, saved):
    store.save_summary(summary_for(saved.session_id))
    items = store.get_action_items(saved.session_id)
    assert [i.text for i in items] == ["hablar con el banco"]
    assert items[0].owner == "Yo"


def test_action_items_deduplicate_on_the_folded_fingerprint(store, saved):
    """The same commitment detected live and again in the final summary must land
    once, even with different accents and punctuation."""
    assert store.add_action_item(saved.session_id, ActionItem("Enviar el informe.", owner="María"))
    assert not store.add_action_item(
        saved.session_id, ActionItem("enviar el informe", owner="Maria")
    )
    assert len(store.get_action_items(saved.session_id)) == 1


def test_no_summary_yet(store, saved):
    assert store.get_summary(saved.session_id) is None


# ------------------------------------------------------------------ plumbing


def test_schema_version_is_stamped(tmp_path):
    path = tmp_path / "v.db"
    with SqliteStore(path):
        pass
    import sqlite3

    with sqlite3.connect(path) as db:
        assert int(db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION


def test_a_future_schema_is_refused_rather_than_corrupted(tmp_path):
    path = tmp_path / "future.db"
    import sqlite3

    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION + 5}")
    with pytest.raises(StoreError, match="newer"):
        SqliteStore(path)


def test_reopening_an_existing_database_is_idempotent(tmp_path, session):
    path = tmp_path / "again.db"
    with SqliteStore(path) as store:
        store.create_session(session)
    with SqliteStore(path) as store:
        assert store.get_session(session.session_id) is not None
        assert store.stats()["sessions"] == 1


def test_wal_mode_is_actually_on(store):
    mode = store._db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_sessions_older_than_finds_the_expired_ones(store, session):
    old = dataclasses.replace(
        session, session_id="old", started_at=dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
    )
    store.create_session(old)
    store.create_session(session)
    cutoff = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    assert [s.session_id for s in store.sessions_older_than(cutoff)] == ["old"]
