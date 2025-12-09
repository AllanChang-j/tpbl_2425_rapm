import argparse
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from collections import Counter

# ---------------------------
# API endpoints
# ---------------------------
BROADCAST_API = "https://api.tpbl.basketball/api/games/{gid}/broadcasts"
STATS_API = "https://api.tpbl.basketball/api/games/{gid}/stats"

# ---------------------------
# event type groups
# ---------------------------
SHOT_TYPES = {"TwoPointer", "ThreePointer"}
SHOT_OUTCOMES = {"Made", "Missed", "AndOne"}
FT_TYPE = "FreeThrow"
FT_OUTCOMES = {"Made", "Missed"}
REB_TYPE = "Rebound"
REB_OUTCOMES = {"Defensive", "Offensive"}
TOV_TYPE = "Turnover"

FOUL_TYPE_ROSTER = "RosterFoul"
META_TYPES = {"Assist", "Block", "Steal", "Timeout", "RosterFoul", "CrewFoul"}

BALL_EVENTS = {"TwoPointer", "ThreePointer", "FreeThrow", "Turnover"}


def ev_str(x):
    return (x or "").strip()


def ev_type(e):
    return ev_str(e.get("type") or e.get("event_type"))


def ev_outcome(e):
    return ev_str(e.get("outcome") or e.get("event_outcome"))


def is_rotation_row(r):
    return ev_type(r) == "Rotation" and ev_outcome(r) in {"Entering", "Leaving"}


def is_shot_row(r):
    return ev_type(r) in SHOT_TYPES and ev_outcome(r) in SHOT_OUTCOMES


def is_free_throw_row(r):
    return ev_type(r) == FT_TYPE and ev_outcome(r) in FT_OUTCOMES


def is_rebound_row(r):
    return ev_type(r) == REB_TYPE and ev_outcome(r) in REB_OUTCOMES


def rebound_is_offensive_row(r):
    return ev_type(r) == REB_TYPE and ev_outcome(r) == "Offensive"


def rebound_is_defensive_row(r):
    return ev_type(r) == REB_TYPE and ev_outcome(r) == "Defensive"


def is_turnover_row(r):
    return ev_type(r) == TOV_TYPE


def is_meta_row(r):
    return ev_type(r) in META_TYPES


def is_offensive_foul_row(r):
    t, o = ev_type(r), ev_outcome(r)
    if t == FOUL_TYPE_ROSTER and ("Offensive" in o):
        return True
    return (t == TOV_TYPE) and (o == "OffensiveFoul" or "OffensiveFoul" in o)


def is_ball_event_row(r):
    return (ev_type(r) in BALL_EVENTS) or is_offensive_foul_row(r)


# ---------------------------
# API fetch
# ---------------------------
def fetch_broadcast(gid, timeout=20):
    r = requests.get(BROADCAST_API.format(gid=gid), timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_stats(gid, timeout=20):
    """Fetch official stats for team totals & per quarter comparison & boxscore/minutes."""
    r = requests.get(STATS_API.format(gid=gid), timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------
# helpers: team / player map
# ---------------------------
def infer_team_ids(df_events):
    """從 events 中推 home_id / away_id。"""
    snap = df_events.dropna(subset=["score_home_id", "score_away_id"]).tail(1)
    if not snap.empty:
        home_id = int(snap["score_home_id"].iloc[0])
        away_id = int(snap["score_away_id"].iloc[0])
    else:
        teams = [t for t in df_events["team_id"].dropna().unique().tolist()]
        if len(teams) >= 2:
            home_id, away_id = teams[0], teams[1]
        else:
            home_id = away_id = None
    return home_id, away_id


def build_player_name_map(df_events):
    pid_to_name = {}
    for _, r in df_events.iterrows():
        pid, nm = r.get("player_id"), r.get("player_name")
        if pd.notna(pid) and pid and isinstance(nm, str) and nm:
            pid_to_name[int(pid)] = nm
    return pid_to_name


# ---------------------------
# flatten rounds.events (PBP)
# ---------------------------
def flatten_round_events(b):
    """
    將 broadcasts['rounds'][*]['events'] 攤平成 DataFrame。

    """
    rows = []
    for rnd in b.get("rounds", []):
        for e in rnd.get("events", []):
            player = e.get("player") or {}
            team = e.get("team") or {}
            cp = e.get("current_points") or {}
            cp_h = (cp.get("home_team") or {})
            cp_a = (cp.get("away_team") or {})

            rows.append({
                "order": e.get("order"),
                "quarter": e.get("quarter"),
                # 原本就有的 event_quarter_time（剩餘毫秒）
                "tms": e.get("event_quarter_time"),
                "event_quarter_time_ms": e.get("event_quarter_time"),
                # 新增：事件建立時間
                "created_at": e.get("created_at"),
                "type": e.get("event_type"),
                "outcome": e.get("event_outcome"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "score_home_id": cp_h.get("id"),
                "score_home_name": cp_h.get("name"),
                "score_home_pts": cp_h.get("points"),
                "score_away_id": cp_a.get("id"),
                "score_away_name": cp_a.get("name"),
                "score_away_pts": cp_a.get("points"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("order").reset_index(drop=True)
    return df


# ---------------------------
# flatten score events
# ---------------------------
def flatten_score_events(b):
    rows = []
    for se in b.get("score_events", []):
        t = ev_str(se.get("event_type"))
        made = bool(se.get("is_made"))
        if not made:
            pts = 0
        else:
            if t == "ThreePointer":
                pts = 3
            elif t == "TwoPointer":
                pts = 2
            elif t == "FreeThrow":
                pts = 1
            else:
                pts = 0

        team = se.get("team") or {}
        rows.append({
            "order": se.get("order"),
            "quarter": se.get("round"),
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "points": pts
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("order").reset_index(drop=True)
    return df


# ---------------------------
# segmentation of possessions
# ---------------------------
def segment_possessions(df_events):
    poss = []
    cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
    rows = df_events.reset_index().to_dict(orient="records")
    i, N = 0, len(rows)

    def start_new_from(r):
        return dict(events=[r["index"]],
                    quarter=int(r["quarter"]),
                    start_idx=r["index"],
                    end_idx=None)

    while i < N:
        e = rows[i]
        # quarter change → force close possession
        if cur["quarter"] is not None and int(e["quarter"]) != cur["quarter"]:
            cur["end_idx"] = cur["end_idx"] or cur["start_idx"]
            poss.append(cur)
            cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
            continue

        # no current possession yet → only start on ball event
        if cur["quarter"] is None:
            if not is_ball_event_row(e):
                i += 1
                continue
            cur["quarter"] = int(e["quarter"])
            cur["start_idx"] = e["index"]

        cur["events"].append(e["index"])

        # turnover or offensive foul → possession ends
        if is_turnover_row(e) or is_offensive_foul_row(e):
            cur["end_idx"] = e["index"]
            poss.append(cur)
            cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
            i += 1
            continue

        # field goal
        if is_shot_row(e):
            outc = ev_outcome(e)
            # And-one: shot + FT + possible rebound
            if outc == "AndOne":
                j = i + 1
                while j < N and is_free_throw_row(rows[j]):
                    cur["events"].append(rows[j]["index"])
                    j += 1
                if j < N and is_rebound_row(rows[j]):
                    if rebound_is_offensive_row(rows[j]):
                        cur["end_idx"] = rows[j]["index"]
                        poss.append(cur)
                        cur = start_new_from(rows[j])
                        i = j + 1
                        continue
                    else:
                        cur["end_idx"] = rows[j]["index"]
                        poss.append(cur)
                        cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
                        i = j + 1
                        continue
                else:
                    end_row = rows[min(j - 1, i)]
                    cur["end_idx"] = end_row["index"]
                    poss.append(cur)
                    cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
                    i = j
                    continue
            else:
                # made shot
                if outc == "Made":
                    j = i + 1
                    consumed = False
                    while j < N:
                        et = ev_type(rows[j])
                        eo = ev_outcome(rows[j])
                        if et in {"RosterFoul", "CrewFoul"} and (
                                "Technical" in eo or "Unsportsmanlike" in eo):
                            cur["events"].append(rows[j]["index"])
                            j += 1
                            consumed = True
                            continue
                        if is_free_throw_row(rows[j]):
                            cur["events"].append(rows[j]["index"])
                            j += 1
                            consumed = True
                            continue
                        break
                    end_row = rows[j - 1] if consumed else e
                    cur["end_idx"] = end_row["index"]
                    poss.append(cur)
                    cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
                    i = j
                    continue

                # missed shot
                if outc == "Missed":
                    j = i + 1
                    while j < N and not (
                        is_rebound_row(rows[j]) or
                        is_turnover_row(rows[j]) or
                        is_shot_row(rows[j]) or
                        is_free_throw_row(rows[j])
                    ):
                        if is_meta_row(rows[j]):
                            cur["events"].append(rows[j]["index"])
                            j += 1
                        else:
                            break
                    if j < N and is_rebound_row(rows[j]):
                        if rebound_is_offensive_row(rows[j]):
                            cur["end_idx"] = rows[j]["index"]
                            poss.append(cur)
                            cur = start_new_from(rows[j])
                            i = j + 1
                            continue
                        else:
                            cur["end_idx"] = rows[j]["index"]
                            poss.append(cur)
                            cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
                            i = j + 1
                            continue
                    else:
                        cur["end_idx"] = e["index"]
                        poss.append(cur)
                        cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
                        i = j
                        continue

                i += 1
                continue

        # pure FT trip
        if is_free_throw_row(e):
            j = i
            while j < N and is_free_throw_row(rows[j]):
                cur["events"].append(rows[j]["index"])
                j += 1
            if j < N:
                nxt = rows[j]
                if is_rebound_row(nxt):
                    if rebound_is_offensive_row(nxt):
                        cur["end_idx"] = nxt["index"]
                        poss.append(cur)
                        cur = start_new_from(nxt)
                        i = j + 1
                        continue
                    else:
                        cur["end_idx"] = nxt["index"]
                        poss.append(cur)
                        cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
                        i = j + 1
                        continue
                else:
                    cur["end_idx"] = rows[j - 1]["index"]
                    poss.append(cur)
                    cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
                    i = j
                    continue
            else:
                cur["end_idx"] = rows[j - 1]["index"]
                poss.append(cur)
                cur = dict(events=[], quarter=None, start_idx=None, end_idx=None)
                i = j
                continue

        i += 1

    if cur["events"]:
        cur["end_idx"] = cur["end_idx"] or cur["start_idx"]
        poss.append(cur)

    return poss


# ---------------------------
# build stints
# ---------------------------
def build_stints(df_events, df_scores, league, season, game_id,
                 filter_5v5=False, merge_adjacent=True):

    home_id, away_id = infer_team_ids(df_events)

    # 1) segment possessions
    poss = segment_possessions(df_events)
    print(f"[debug] possessions: {len(poss)} from events: {len(df_events)}")
    print("[debug] poss per quarter:", dict(Counter([p["quarter"] for p in poss])))

    # 2) build name map
    pid_to_name = build_player_name_map(df_events)

    # 3) track rotations → lineup_at
    on_home, on_away = set(), set()
    lineup_at = {}
    for idx, r in df_events.iterrows():
        if is_rotation_row(r):
            pid = r.get("player_id")
            tid = r.get("team_id")
            outc = ev_outcome(r)
            if pd.notna(pid) and pid:
                pid = int(pid)
                if outc == "Entering":
                    if tid == home_id:
                        on_home.add(pid)
                    elif tid == away_id:
                        on_away.add(pid)
                elif outc == "Leaving":
                    on_home.discard(pid)
                    on_away.discard(pid)
        lineup_at[idx] = (tuple(sorted(on_home)), tuple(sorted(on_away)))

    # 4) build possessions ranges
    poss_by_q = {}
    for k, p in enumerate(poss):
        q = int(p["quarter"])
        beg_idx, end_idx = p["start_idx"], p["end_idx"]
        so = int(df_events.loc[beg_idx, "order"])
        eo = int(df_events.loc[end_idx, "order"])
        if so > eo:
            so, eo = eo, so
        poss_by_q.setdefault(q, []).append({"k": k, "so": so, "eo": eo})
    for q in poss_by_q:
        poss_by_q[q].sort(key=lambda x: (x["so"], x["eo"]))

    # 5) allocate score events to possessions
    poss_points = [{"h": 0.0, "a": 0.0} for _ in poss]
    uncovered = []

    if not df_scores.empty:
        for q, seg in df_scores.groupby(df_scores["quarter"].astype(int)):
            seg = seg.sort_values("order")
            ranges = poss_by_q.get(q, [])
            i = 0
            for _, se in seg.iterrows():
                pts = int(se["points"])
                if pts <= 0:
                    continue
                ord_k = int(se["order"])
                while i < len(ranges) and ranges[i]["eo"] < ord_k:
                    i += 1
                if i >= len(ranges):
                    uncovered.append(se)
                    continue
                if ranges[i]["so"] <= ord_k <= ranges[i]["eo"]:
                    k = ranges[i]["k"]
                    tid = int(se["team_id"]) if pd.notna(se["team_id"]) else None
                    if tid == home_id:
                        poss_points[k]["h"] += pts
                    elif tid == away_id:
                        poss_points[k]["a"] += pts
                else:
                    uncovered.append(se)

    # 6) attach uncovered score events to nearest possession
    if uncovered:
        print(f"[audit] uncovered score_events (first 10 of {len(uncovered)}):")
        for r in uncovered[:10]:
            print(f"  - order={int(r['order'])} round={int(r['quarter'])} pts={int(r['points'])}")

        left_edges = np.array([int(df_events.loc[p['start_idx'], "order"])
                               for p in poss], dtype=np.int64)
        for _, se in enumerate(uncovered):
            ord_k = int(se["order"])
            pos = int(np.searchsorted(left_edges, ord_k, side="left"))
            if pos < len(poss):
                target = pos
            elif pos > 0:
                target = pos - 1
            else:
                continue

            pts = int(se["points"])
            tid = int(se["team_id"]) if pd.notna(se["team_id"]) else None
            if tid == home_id:
                poss_points[target]["h"] += pts
            elif tid == away_id:
                poss_points[target]["a"] += pts

    # 7) build output rows (stints)
    rows = []
    for idx_p, p in enumerate(poss):
        beg, end = p["start_idx"], p["end_idx"]
        q = int(p["quarter"])
        start_ms = int(df_events.loc[beg, "tms"]) if pd.notna(df_events.loc[beg, "tms"]) else 0
        end_ms = int(df_events.loc[end, "tms"]) if pd.notna(df_events.loc[end, "tms"]) else start_ms

        home_ids, away_ids = lineup_at.get(beg, (tuple(), tuple()))
        hpts = poss_points[idx_p]["h"]
        apts = poss_points[idx_p]["a"]

        start_sec = start_ms // 1000
        end_sec = end_ms // 1000
        duration_sec = abs(end_sec - start_sec)  # 用剩餘秒數差估算時間長度

        rows.append({
            "league": league,
            "season": season,
            "game_id": str(game_id),
            "period": q,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": float(duration_sec),
            "home_players": tuple((pid_to_name.get(pid, ""), pid) for pid in home_ids),
            "away_players": tuple((pid_to_name.get(pid, ""), pid) for pid in away_ids),
            "home_player_ids": tuple(home_ids),
            "away_player_ids": tuple(away_ids),
            "home_pts": float(hpts),
            "away_pts": float(apts),
            "possessions": 1.0,
            "net_points": float(hpts - apts),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # filter 5v5
    if filter_5v5:
        out = out[(out["home_player_ids"].apply(len) == 5) &
                  (out["away_player_ids"].apply(len) == 5)].reset_index(drop=True)

    # merge adjacent stints
    if merge_adjacent and not out.empty:
        merged = []
        cur = out.iloc[0].to_dict()
        for r in out.iloc[1:].itertuples(index=False):
            same = (tuple(cur["home_player_ids"]) == tuple(r.home_player_ids)) and \
                   (tuple(cur["away_player_ids"]) == tuple(r.away_player_ids)) and \
                   (cur["period"] == r.period)
            consecutive = (abs(int(cur["end_sec"]) - int(r.start_sec)) <= 1)
            if same and consecutive:
                cur["end_sec"] = int(r.end_sec)
                cur["duration_sec"] += float(r.duration_sec)
                cur["home_pts"] += float(r.home_pts)
                cur["away_pts"] += float(r.away_pts)
                cur["possessions"] += float(r.possessions)
                cur["net_points"] = float(cur["home_pts"] - float(cur["away_pts"]))
            else:
                merged.append(cur)
                cur = r._asdict()
        merged.append(cur)
        out = pd.DataFrame(merged)

    return out.reset_index(drop=True)


# ---------------------------
# PBP-derived minutes & boxscore
# ---------------------------

def compute_pbp_minutes_from_rotations(df_events):
    """
    用 Rotation 事件 + event_quarter_time (tms, 該節剩餘毫秒)
    來計算每個球員的上場時間（秒）。

    """
    if df_events.empty:
        return pd.DataFrame()

    df = df_events.copy()
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")
    df["tms"] = pd.to_numeric(df["tms"], errors="coerce")

    df_rot = df[df.apply(is_rotation_row, axis=1)].copy()
    if df_rot.empty:
        print("[WARN] compute_pbp_minutes_from_rotations: no Rotation rows found")
        return pd.DataFrame()

    df_rot = df_rot.sort_values(["quarter", "order"])

    minutes = {}

    for q, seg in df_rot.groupby("quarter"):
        seg = seg.sort_values("order")
        if seg["tms"].notna().sum() == 0:
            continue

        q_end_ms = 0.0

        on_court = {}

        for _, r in seg.iterrows():
            tms = r["tms"]
            if pd.isna(tms):
                continue

            pid = r.get("player_id")
            tid = r.get("team_id")
            if pd.isna(pid) or pd.isna(tid):
                continue
            pid = int(pid)
            tid = int(tid)
            key = (tid, pid)

            outcome = ev_outcome(r)  # Entering / Leaving
            if outcome == "Entering":
                on_court[key] = tms
            elif outcome == "Leaving":
                if key in on_court:
                    enter_ms = on_court[key]
                    dur_ms = float(enter_ms) - float(tms)
                    if dur_ms < 0:
                        dur_ms = 0.0
                    sec = dur_ms / 1000.0

                    player_name = r.get("player_name") or ""
                    mkey = (tid, pid, player_name)
                    minutes[mkey] = minutes.get(mkey, 0.0) + sec

                    del on_court[key]
                else:
                    pass

        for (tid, pid), enter_ms in on_court.items():
            dur_ms = float(enter_ms) - float(q_end_ms)
            if dur_ms < 0:
                dur_ms = 0.0
            sec = dur_ms / 1000.0

            name_series = df_events.loc[
                (df_events["team_id"] == tid) & (df_events["player_id"] == pid),
                "player_name"
            ]
            player_name = name_series.dropna().iloc[0] if not name_series.dropna().empty else ""

            mkey = (tid, pid, player_name)
            minutes[mkey] = minutes.get(mkey, 0.0) + sec

    rows = []
    for (tid, pid, name), sec in minutes.items():
        rows.append({
            "team_id": tid,
            "player_id": pid,
            "player_name": name,
            "sec_played": sec,
            "min_played": sec / 60.0,
        })

    df_min = pd.DataFrame(rows)
    return df_min


def compute_pbp_minutes_from_stints(st_df, df_events):
    """
    用 stints 的 duration_sec + lineup 計算每名球員的上場秒數。
    """
    if st_df.empty:
        return pd.DataFrame()

    home_id, away_id = infer_team_ids(df_events)
    pid_to_name = build_player_name_map(df_events)

    rows = []
    for _, r in st_df.iterrows():
        dur = float(r.get("duration_sec", 0.0))
        # home
        for pid in r["home_player_ids"]:
            pid = int(pid)
            rows.append({
                "team_id": home_id,
                "player_id": pid,
                "player_name": pid_to_name.get(pid, ""),
                "sec_played": dur
            })
        # away
        for pid in r["away_player_ids"]:
            pid = int(pid)
            rows.append({
                "team_id": away_id,
                "player_id": pid,
                "player_name": pid_to_name.get(pid, ""),
                "sec_played": dur
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.groupby(["team_id", "player_id", "player_name"], as_index=False)
            .agg({"sec_played": "sum"}))
    df["min_played"] = df["sec_played"] / 60.0
    return df

def build_pbp_boxscore_by_period(df_events, st_df=None):
    """
    依據 PBP 事件產生「每節 per-player boxscore」：
      key = (team_id, player_id, period)

    - 從事件算 PTS / FGM / FGA / 3PM / 3PA / FTM / FTA / REB / AST / TOV / PF / STL / BLK
    - 若提供 st_df，會額外用 stints 算每節的「+/-」
    """
    if df_events.empty:
        return pd.DataFrame()

    pid_to_name = build_player_name_map(df_events)

    stats = {}
    for _, r in df_events.iterrows():
        pid = r.get("player_id")
        tid = r.get("team_id")
        q = r.get("quarter")

        if pd.isna(pid) or pd.isna(tid) or pd.isna(q):
            continue
        pid = int(pid)
        tid = int(tid)
        try:
            q = int(q)
        except Exception:
            continue

        key = (tid, pid, q)
        if key not in stats:
            stats[key] = {
                "team_id": tid,
                "player_id": pid,
                "player_name": pid_to_name.get(pid, ""),
                "period": q,
                "PTS": 0,
                "FGM": 0,
                "FGA": 0,
                "3PM": 0,
                "3PA": 0,
                "FTM": 0,
                "FTA": 0,
                "OREB": 0,
                "DREB": 0,
                "AST": 0,
                "TOV": 0,
                "PF": 0,
                "STL": 0,
                "BLK": 0,
                "+/-": 0.0,
            }

        t = ev_type(r)
        o = ev_outcome(r)

        # field goals
        if t in SHOT_TYPES and o in SHOT_OUTCOMES:
            stats[key]["FGA"] += 1
            pts = 2 if t == "TwoPointer" else 3
            if o in {"Made", "AndOne"}:
                stats[key]["FGM"] += 1
                stats[key]["PTS"] += pts
                if t == "ThreePointer":
                    stats[key]["3PM"] += 1
                    stats[key]["3PA"] += 1
                else:
                    stats[key]["3PA"] += 0
            else:
                if t == "ThreePointer":
                    stats[key]["3PA"] += 1

        # free throws
        if t == FT_TYPE and o in FT_OUTCOMES:
            stats[key]["FTA"] += 1
            if o == "Made":
                stats[key]["FTM"] += 1
                stats[key]["PTS"] += 1

        # rebounds
        if t == REB_TYPE and o in REB_OUTCOMES:
            if o == "Offensive":
                stats[key]["OREB"] += 1
            elif o == "Defensive":
                stats[key]["DREB"] += 1

        # assists
        if ev_type(r) == "Assist":
            stats[key]["AST"] += 1

        # turnovers
        if ev_type(r) == TOV_TYPE:
            stats[key]["TOV"] += 1

        # fouls
        if ev_type(r) == FOUL_TYPE_ROSTER:
            stats[key]["PF"] += 1

        # steals
        if ev_type(r) == "Steal":
            stats[key]["STL"] += 1

        # blocks
        if ev_type(r) == "Block":
            stats[key]["BLK"] += 1

    # 用 stints 算每節 per-player +/-
    if st_df is not None and not st_df.empty:
        home_id, away_id = infer_team_ids(df_events)

        def ensure_row(team_id, pid, period):
            key = (team_id, pid, period)
            if key not in stats:
                stats[key] = {
                    "team_id": team_id,
                    "player_id": pid,
                    "player_name": pid_to_name.get(pid, ""),
                    "period": period,
                    "PTS": 0,
                    "FGM": 0,
                    "FGA": 0,
                    "3PM": 0,
                    "3PA": 0,
                    "FTM": 0,
                    "FTA": 0,
                    "OREB": 0,
                    "DREB": 0,
                    "AST": 0,
                    "TOV": 0,
                    "PF": 0,
                    "STL": 0,
                    "BLK": 0,
                    "+/-": 0.0,
                }
            return key

        for _, s in st_df.iterrows():
            period = int(s.get("period", 0))
            net = float(s.get("net_points", 0.0))

            # home players
            for pid in s["home_player_ids"]:
                pid = int(pid)
                key = ensure_row(home_id, pid, period)
                stats[key]["+/-"] += net

            # away players
            for pid in s["away_player_ids"]:
                pid = int(pid)
                key = ensure_row(away_id, pid, period)
                stats[key]["+/-"] -= net

    box_q = pd.DataFrame(list(stats.values()))
    return box_q

def build_pbp_boxscore(df_events, st_df):
    """
    依據 PBP 事件產生 per-player boxscore：
      - PTS, FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, AST, TOV, PF, STL, BLK, +/-
    並附上 team_id。
    """
    if df_events.empty:
        return pd.DataFrame()

    pid_to_name = build_player_name_map(df_events)

    stats = {}
    for _, r in df_events.iterrows():
        pid = r.get("player_id")
        tid = r.get("team_id")
        if pd.isna(pid) or pd.isna(tid):
            continue
        pid = int(pid)
        tid = int(tid)
        key = (tid, pid)
        if key not in stats:
            stats[key] = {
                "team_id": tid,
                "player_id": pid,
                "player_name": pid_to_name.get(pid, ""),
                "PTS": 0,
                "FGM": 0,
                "FGA": 0,
                "3PM": 0,
                "3PA": 0,
                "FTM": 0,
                "FTA": 0,
                "OREB": 0,
                "DREB": 0,
                "AST": 0,
                "TOV": 0,
                "PF": 0,
                "STL": 0,
                "BLK": 0,
                "+/-": 0.0,
            }

        t = ev_type(r)
        o = ev_outcome(r)

        # field goals
        if t in SHOT_TYPES and o in SHOT_OUTCOMES:
            stats[key]["FGA"] += 1
            pts = 2 if t == "TwoPointer" else 3
            if o in {"Made", "AndOne"}:
                stats[key]["FGM"] += 1
                stats[key]["PTS"] += pts
                if t == "ThreePointer":
                    stats[key]["3PM"] += 1
                    stats[key]["3PA"] += 1
                else:
                    stats[key]["3PA"] += 0
            else:
                if t == "ThreePointer":
                    stats[key]["3PA"] += 1

        # free throws
        if t == FT_TYPE and o in FT_OUTCOMES:
            stats[key]["FTA"] += 1
            if o == "Made":
                stats[key]["FTM"] += 1
                stats[key]["PTS"] += 1

        # rebounds
        if t == REB_TYPE and o in REB_OUTCOMES:
            if o == "Offensive":
                stats[key]["OREB"] += 1
            elif o == "Defensive":
                stats[key]["DREB"] += 1

        # assists
        if ev_type(r) == "Assist":
            stats[key]["AST"] += 1

        # turnovers
        if ev_type(r) == TOV_TYPE:
            stats[key]["TOV"] += 1

        # fouls
        if ev_type(r) == FOUL_TYPE_ROSTER:
            stats[key]["PF"] += 1

        # steals
        if ev_type(r) == "Steal":
            stats[key]["STL"] += 1

        # blocks
        if ev_type(r) == "Block":
            stats[key]["BLK"] += 1

    # 用 stints 算全場 per-player +/-
    home_id, away_id = infer_team_ids(df_events)
    plusminus = {}

    if st_df is not None and not st_df.empty:
        for _, s in st_df.iterrows():
            net = float(s.get("net_points", 0.0))  # home_pts - away_pts

            # home 球員：+net
            for pid in s["home_player_ids"]:
                pid = int(pid)
                key_pm = (home_id, pid)
                plusminus[key_pm] = plusminus.get(key_pm, 0.0) + net

            # away 球員：-net
            for pid in s["away_player_ids"]:
                pid = int(pid)
                key_pm = (away_id, pid)
                plusminus[key_pm] = plusminus.get(key_pm, 0.0) - net

    for key_pm, pm in plusminus.items():
        if key_pm in stats:
            stats[key_pm]["+/-"] = pm

    box = pd.DataFrame(list(stats.values()))

    # minutes: merge from rotations
    mins = compute_pbp_minutes_from_rotations(df_events)
    if not mins.empty:
        box = box.merge(
            mins[["team_id", "player_id", "sec_played", "min_played"]],
            on=["team_id", "player_id"],
            how="left"
        )
    else:
        box["sec_played"] = np.nan
        box["min_played"] = np.nan

    return box

# ---------------------------
# 官方 stats → boxscore & minutes
# ---------------------------
def _parse_time_to_seconds(t):
    """
    將 "MM:SS" 或 "HH:MM:SS" 或 int 秒數 轉成秒。
    """
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    if isinstance(t, (int, float)):
        return int(t)
    if isinstance(t, str):
        parts = t.split(":")
        try:
            parts = [int(x) for x in parts]
        except ValueError:
            return None
        if len(parts) == 2:
            m, s = parts
            return m * 60 + s
        if len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
    return None

def flatten_stats_boxscore_by_period(stats_json):
    """
    官方 stats 每節 per-player boxscore：

      stats_json["home_team"]["players"]["rounds"][round_key][roster_id]
      stats_json["away_team"]["players"]["rounds"][round_key][roster_id]

    round_key: "1", "2", "3", "4"（或 int）
    """
    rows = []

    if not isinstance(stats_json, dict):
        print("[DEBUG] flatten_stats_boxscore_by_period: stats_json is not dict, type =", type(stats_json))
        return pd.DataFrame()

    for side in ["home_team", "away_team"]:
        team_block = stats_json.get(side)
        if not isinstance(team_block, dict):
            print(f"[DEBUG] flatten_stats_boxscore_by_period: '{side}' missing or not dict")
            continue

        team_id = team_block.get("id")
        team_name = team_block.get("name")
        print(f"[DEBUG] [per-period] processing {side}: team_id={team_id}, name={team_name}")

        players_block = team_block.get("players") or {}
        rounds_block = players_block.get("rounds") or {}

        if not isinstance(rounds_block, dict) or not rounds_block:
            print(f"[DEBUG] [per-period] team_id={team_id} has no players.rounds")
            continue

        for rk, players_in_round in rounds_block.items():
            # 跳過 total，只要 1~4 節
            rk_str = str(rk)
            if not rk_str.isdigit():
                continue
            period = int(rk_str)
            if not isinstance(players_in_round, dict):
                continue

            cnt = 0
            for roster_id, p in players_in_round.items():
                if not isinstance(p, dict):
                    continue

                status = p.get("status")
                # 如果不想排除 WITHDRAWN，可以拿掉這個 if
                #if status == "WITHDRAWN":
                #    continue

                pid = p.get("id")
                name = p.get("name")
                number = p.get("number")
                gohoops_roster_id = p.get("gohoops_roster_id") or roster_id
                gohoops_id = p.get("gohoops_id")

                pts = p.get("score")
                fgm = p.get("field_goals_made")
                fga = p.get("field_goals_attempted")
                tpm = p.get("three_pointers_made")
                tpa = p.get("three_pointers_attempted")
                ftm = p.get("free_throws_made")
                fta = p.get("free_throws_attempted")
                oreb = p.get("offensive_rebounds")
                dreb = p.get("defensive_rebounds")
                ast = p.get("assists")
                tov = p.get("turnovers")
                pf = p.get("fouls") or p.get("personal_fouls")
                stl = p.get("steals")
                blk = p.get("blocks")
                plus_minus = p.get("plus_minus") or p.get("plusminus")

                raw_sec = p.get("time_on_court")
                sec = _parse_time_to_seconds(raw_sec)

                rows.append({
                    "team_id": team_id,
                    "player_id": pid,
                    "player_name": name,
                    "number": number,
                    "period": period,
                    "gohoops_roster_id": gohoops_roster_id,
                    "gohoops_id": gohoops_id,
                    "PTS": pts,
                    "FGM": fgm,
                    "FGA": fga,
                    "3PM": tpm,
                    "3PA": tpa,
                    "FTM": ftm,
                    "FTA": fta,
                    "OREB": oreb,
                    "DREB": dreb,
                    "AST": ast,
                    "TOV": tov,
                    "PF": pf,
                    "STL": stl,
                    "BLK": blk,
                    "+/-": plus_minus,
                    "sec_played": sec,
                    "min_played": sec / 60.0 if sec is not None else None,
                })
                cnt += 1

            print(f"[DEBUG] [per-period] team_id={team_id}, round={period} players={cnt}")

    df = pd.DataFrame(rows)
    print("[DEBUG] flatten_stats_boxscore_by_period: rows =", len(df))
    return df

def flatten_stats_players(stats_json):
    """
    依照 TPBL 實際結構，硬抓：
      stats_json["home_team"]["players"]["total"]
      stats_json["away_team"]["players"]["total"]

    total_block 形態：
      {
        "126317": {
            "id": 35,
            "name": "李睿麒",
            "score": 7,
            "time_on_court": 463,
            ...
        },
        ...
      }
    """
    players = []

    if not isinstance(stats_json, dict):
        print("[DEBUG] flatten_stats_players: stats_json is not dict, type =", type(stats_json))
        return players

    for side in ["home_team", "away_team"]:
        team_block = stats_json.get(side)
        if not isinstance(team_block, dict):
            print(f"[DEBUG] flatten_stats_players: '{side}' not in stats_json or not dict")
            continue

        team_id = team_block.get("id")
        team_name = team_block.get("name")
        print(f"[DEBUG] processing {side}: team_id={team_id}, name={team_name}")

        players_block = team_block.get("players") or {}

        total_block = players_block.get("total") or {}

        if not isinstance(total_block, dict) or not total_block:
            print(f"[DEBUG] team_id={team_id} players.total is empty or not dict")
            continue

        cnt = 0
        for roster_id, p in total_block.items():
            if not isinstance(p, dict):
                continue
            row = dict(p)
            row["_team_id"] = team_id
            row["_gohoops_roster_id"] = roster_id
            players.append(row)
            cnt += 1

        print(f"[DEBUG] team_id={team_id} total players collected = {cnt}")

    print("[DEBUG] flatten_stats_players: total players =", len(players))
    return players

def flatten_stats_boxscore(stats_json):
    """
    依照 TPBL stats 結構，產生 per-player boxscore + minutes。
    一定回傳 pandas.DataFrame（就算沒有資料，也會是空的 DataFrame）
    """
    players = flatten_stats_players(stats_json)
    if not players:
        print("[DEBUG] flatten_stats_boxscore: no players found, return empty DataFrame")
        return pd.DataFrame()

    rows = []
    for p in players:
        status = p.get("status")
        # 如果你想保留 WITHDRAWN，就把這兩行註解掉
        #if status == "WITHDRAWN":
            #continue

        team_id = p.get("_team_id")

        pid = p.get("id")  # 這個在 sample 裡是 35, 36... 跟 pbp 的 player_id 一致
        name = p.get("name")
        number = p.get("number")
        gohoops_roster_id = p.get("gohoops_roster_id") or p.get("_gohoops_roster_id")
        gohoops_id = p.get("gohoops_id")

        pts = p.get("score")
        fgm = p.get("field_goals_made")
        fga = p.get("field_goals_attempted")
        tpm = p.get("three_pointers_made")
        tpa = p.get("three_pointers_attempted")
        ftm = p.get("free_throws_made")
        fta = p.get("free_throws_attempted")
        oreb = p.get("offensive_rebounds")
        dreb = p.get("defensive_rebounds")
        ast = p.get("assists")
        tov = p.get("turnovers")
        pf = p.get("fouls") or p.get("personal_fouls")

        # 新增：抄截 / 阻攻 / +/-（名稱可以再對照實際 JSON）
        stl = p.get("steals")
        blk = p.get("blocks")
        plus_minus = p.get("plus_minus") or p.get("plusminus")

        raw_sec = p.get("time_on_court")  # sample 裡是 463 這樣的秒數
        sec = _parse_time_to_seconds(raw_sec)

        rows.append({
            "team_id": team_id,
            "player_id": pid,
            "player_name": name,
            "number": number,
            "gohoops_roster_id": gohoops_roster_id,
            "gohoops_id": gohoops_id,
            "PTS": pts,
            "FGM": fgm,
            "FGA": fga,
            "3PM": tpm,
            "3PA": tpa,
            "FTM": ftm,
            "FTA": fta,
            "OREB": oreb,
            "DREB": dreb,
            "AST": ast,
            "TOV": tov,
            "PF": pf,
            "STL": stl,
            "BLK": blk,
            "+/-": plus_minus,
            "sec_played": sec,
            "min_played": sec / 60.0 if sec is not None else None,
        })

    df = pd.DataFrame(rows)
    print("[DEBUG] flatten_stats_boxscore: rows =", len(df))
    return df


def flatten_stats_minutes(stats_json):
    """
    從官方 stats 抽出 minutes，用來跟 PBP minutes 對比。
    一定回傳 DataFrame。
    """
    df = flatten_stats_boxscore(stats_json)
    if df.empty:
        print("[DEBUG] flatten_stats_minutes: stats_box is empty, return empty DataFrame")
        return pd.DataFrame(columns=["team_id", "player_id", "player_name", "sec_played", "min_played"])
    return df[["team_id", "player_id", "player_name", "sec_played", "min_played"]]
# ---------------------------
# audit with official stats
# ---------------------------
def audit_totals(st, df_e, df_s, stats=None):
    def _last_nonnull(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return int(x.iloc[-1]) if not x.empty else None

    # scoreboard extracted from broadcast
    f_h = _last_nonnull(df_e["score_home_pts"])
    f_a = _last_nonnull(df_e["score_away_pts"])

    home_id, away_id = infer_team_ids(df_e)

    # score_events summary
    ssum = df_s.groupby("team_id")["points"].sum() if not df_s.empty else pd.Series(dtype=int)
    sev_h = int(ssum.get(home_id, 0))
    sev_a = int(ssum.get(away_id, 0))

    # stints summary
    sh = float(st["home_pts"].sum())
    sa = float(st["away_pts"].sum())

    print("========== BASIC CHECK ==========")
    print(f"[broadcast scoreboard]  home={f_h}  away={f_a}")
    print(f"[score_events totals]  home={sev_h}  away={sev_a}")
    print(f"[stints totals]        home={sh}  away={sa}")
    print("=================================")


def audit_boxscore(pbp_box, stats_box, out_prefix="audit_boxscore", append=False):
    """
    比對 pbp_box vs stats_box 的 boxscore。
    - 會嘗試比對 PTS, FGA, FGM, 3PA, 3PM, FTA, FTM, OREB, DREB, AST, TOV, PF, STL, BLK, +/-。
    - 對於「stats 有但 pbp 沒有」的欄位，會列出為尚未實作，不納入 error。
    - mismatch 明細會輸出成 <out_prefix>_mismatch.csv，
      summary 會輸出成 <out_prefix>_summary_by_game.csv / _by_col.csv。
    """
    if pbp_box.empty or stats_box.empty:
        print("[audit_boxscore] pbp_box 或 stats_box 為空，略過 boxscore 比對。")
        return pd.DataFrame()

    # 想要完整檢查的欄位（理想清單）
    cols_all = [
        "PTS",
        "FGA", "FGM",
        "3PA", "3PM",
        "FTA", "FTM",
        "OREB", "DREB",
        "AST", "TOV", "PF",
        "STL", "BLK",
        "+/-",
    ]

    # 實際兩邊都有的欄位
    cols_common = [
        c for c in cols_all
        if (c in pbp_box.columns) and (c in stats_box.columns)
    ]

    # stats 有，但 pbp 目前沒有實作的欄位（例如 STL / BLK / +/-）
    cols_missing_from_pbp = [
        c for c in cols_all
        if (c not in pbp_box.columns) and (c in stats_box.columns)
    ]

    if cols_missing_from_pbp:
        print("[INFO][boxscore] 目前 PBP 尚未實作、因此未納入比對的欄位：")
        print("       " + ", ".join(cols_missing_from_pbp))

    if not cols_common:
        print("[WARN][boxscore] pbp_box 和 stats_box 沒有任何共同的統計欄位可比對。")
        return pd.DataFrame()

    print("[INFO][boxscore] 本次實際比對欄位：", ", ".join(cols_common))

    key_cols = ["team_id", "player_id"]

    # 確保 stats_box 至少有 key + cols_common
    cols_stats = [c for c in key_cols + cols_common if c in stats_box.columns]

    merged = pbp_box.merge(
        stats_box[cols_stats],
        on=key_cols,
        how="outer",
        suffixes=("_pbp", "_stats"),
        indicator=True,
    )

    records = []

    for _, row in merged.iterrows():
        player_name = row.get("player_name", "")
        team_id = row.get("team_id")

        for col in cols_common:
            pbp_val = row.get(f"{col}_pbp")
            stats_val = row.get(f"{col}_stats")

            # 兩邊都 NaN → 略過
            if pd.isna(pbp_val) and pd.isna(stats_val):
                continue

            if pbp_val != stats_val:
                try:
                    diff = pbp_val - stats_val
                except Exception:
                    diff = None

                records.append({
                    "team_id": team_id,
                    "player_id": row.get("player_id"),
                    "player_name": player_name,
                    "col": col,
                    "pbp": pbp_val,
                    "stats": stats_val,
                    "diff": diff,
                })

    if not records:
        print("[OK][boxscore] 所有比對欄位皆與官方 boxscore 一致 ✅")
        return pd.DataFrame()

    mismatch_df = pd.DataFrame(records)
    print(f"[WARN][boxscore] 共發現 {len(mismatch_df)} 筆不一致紀錄")

    # 輸出明細
    mismatch_path = Path(f"{out_prefix}_mismatch.csv")

    if append:
        # 🔹 多場模式：累積寫在同一個檔案
        file_exists = mismatch_path.exists()
        mismatch_df.to_csv(
            mismatch_path,
            mode="a",
            header=not file_exists,  # 第一次有 header，之後就不寫 header
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[boxscore] mismatch 追加寫入 → {mismatch_path}")
        # append 模式先不幫你做 summary（因為要整體算會比較合理）
        return mismatch_df
    else:
        # 🔹 單場模式：跟原本一樣，覆蓋寫入 + summary
        mismatch_df.to_csv(mismatch_path, index=False, encoding="utf-8-sig")
        print(f"[boxscore] mismatch 明細已輸出 → {mismatch_path}")

        by_col = (
            mismatch_df
            .groupby("col")
            .size()
            .reset_index(name="n_mismatch")
            .sort_values("n_mismatch", ascending=False)
        )
        by_col_path = Path(f"{out_prefix}_summary_by_col.csv")
        by_col.to_csv(by_col_path, index=False, encoding="utf-8-sig")
        print(f"[boxscore] mismatch summary(by col) 已輸出 → {by_col_path}")

        by_player = (
            mismatch_df
            .groupby(["team_id", "player_id", "player_name"])
            .size()
            .reset_index(name="n_mismatch")
            .sort_values("n_mismatch", ascending=False)
        )
        by_player_path = Path(f"{out_prefix}_summary_by_player.csv")
        by_player.to_csv(by_player_path, index=False, encoding="utf-8-sig")
        print(f"[boxscore] mismatch summary(by player) 已輸出 → {by_player_path}")




    return mismatch_df

def audit_boxscore_by_period(pbp_box_q, stats_box_q, out_prefix="audit_boxscore_by_period"):
    """
    每節 per-player boxscore 對照。
    - 比對欄位邏輯同 audit_boxscore，但 key 多一個 period。
    """
    if pbp_box_q.empty or stats_box_q.empty:
        print("[audit_boxscore_by_period] pbp_box_q 或 stats_box_q 為空，略過每節 boxscore 比對。")
        return pd.DataFrame()

    cols_all = [
        "PTS",
        "FGA", "FGM",
        "3PA", "3PM",
        "FTA", "FTM",
        "OREB", "DREB",
        "AST", "TOV", "PF",
        "STL", "BLK",
        "+/-",
    ]

    cols_common = [
        c for c in cols_all
        if (c in pbp_box_q.columns) and (c in stats_box_q.columns)
    ]

    cols_missing_from_pbp = [
        c for c in cols_all
        if (c not in pbp_box_q.columns) and (c in stats_box_q.columns)
    ]

    if cols_missing_from_pbp:
        print("[INFO][by-period] 目前 PBP 尚未實作、因此未納入比對的欄位：")
        print("       " + ", ".join(cols_missing_from_pbp))

    if not cols_common:
        print("[WARN][by-period] pbp_box_q 和 stats_box_q 沒有共同統計欄位可比對。")
        return pd.DataFrame()

    print("[INFO][by-period] 本次實際比對欄位：", ", ".join(cols_common))

    key_cols = ["team_id", "player_id", "period"]
    cols_stats = [c for c in key_cols + cols_common if c in stats_box_q.columns]

    merged = pbp_box_q.merge(
        stats_box_q[cols_stats],
        on=key_cols,
        how="outer",
        suffixes=("_pbp", "_stats"),
        indicator=True,
    )

    records = []

    for _, row in merged.iterrows():
        name = row.get("player_name", "")
        team_id = row.get("team_id")
        period = row.get("period")

        for col in cols_common:
            pbp_val = row.get(f"{col}_pbp")
            stats_val = row.get(f"{col}_stats")

            if pd.isna(pbp_val) and pd.isna(stats_val):
                continue

            if pbp_val != stats_val:
                try:
                    diff = pbp_val - stats_val
                except Exception:
                    diff = None

                records.append({
                    "team_id": team_id,
                    "player_id": row.get("player_id"),
                    "player_name": name,
                    "period": period,
                    "col": col,
                    "pbp": pbp_val,
                    "stats": stats_val,
                    "diff": diff,
                })

    if not records:
        print("[OK][by-period] 所有比對欄位皆與官方每節 boxscore 一致 ✅")
        return pd.DataFrame()

    mismatch_df = pd.DataFrame(records)
    print(f"[WARN][by-period] 共發現 {len(mismatch_df)} 筆不一致紀錄")

    mismatch_path = Path(f"{out_prefix}_mismatch.csv")
    mismatch_df.to_csv(mismatch_path, index=False, encoding="utf-8-sig")
    print(f"[by-period] mismatch 明細已輸出 → {mismatch_path}")

    by_col = (
        mismatch_df
        .groupby("col")
        .size()
        .reset_index(name="n_mismatch")
        .sort_values("n_mismatch", ascending=False)
    )
    by_col_path = Path(f"{out_prefix}_summary_by_col.csv")
    by_col.to_csv(by_col_path, index=False, encoding="utf-8-sig")
    print(f"[by-period] mismatch summary(by col) 已輸出 → {by_col_path}")

    return mismatch_df

def audit_minutes(pbp_min, stats_min):
    if pbp_min.empty or stats_min.empty:
        print("[audit_minutes] pbp_min 或 stats_min 為空，略過上場時間比對。")
        return

    merged = pbp_min.merge(
        stats_min,
        on=["team_id", "player_id"],
        how="outer",
        suffixes=("_pbp", "_stats")
    )

    print("========== MINUTES CHECK ==========")
    for _, r in merged.iterrows():
        name = r.get("player_name_pbp") or r.get("player_name_stats") or ""
        sec_pbp = r.get("sec_played_pbp")
        sec_stats = r.get("sec_played_stats")
        if pd.isna(sec_pbp) and pd.isna(sec_stats):
            continue
        # 容許小誤差（例如 2 秒內）
        if (sec_pbp is None) or (sec_stats is None):
            print(f"[WARN] {name} one side missing minutes: pbp={sec_pbp}, stats={sec_stats}")
        else:
            diff = abs(float(sec_pbp) - float(sec_stats))
            if diff > 2.0:
                print(f"[DIFF] {name} (team {r['team_id']}) sec: pbp={sec_pbp}, stats={sec_stats} (diff={diff})")
    print("===================================")


# ---------------------------
# per-game pipeline
# ---------------------------
def process_single_game(game_id, season, league,
                        out_csv_base,
                        filter_5v5=False,
                        merge_adjacent=True):
    print(f"[stints] fetching game {game_id} ...", flush=True)

    # 1) 抓 API
    broadcast = fetch_broadcast(game_id)
    stats = fetch_stats(game_id)

    # 2) 展開 broadcast → events / score_events
    df_e = flatten_round_events(broadcast)
    df_s = flatten_score_events(broadcast)

    if df_e.empty:
        print("[stints] NO events in broadcast. Abort.")
        return

    # 3) 建 stint
    st = build_stints(
        df_e, df_s,
        league=league,
        season=season,
        game_id=str(game_id),
        filter_5v5=filter_5v5,
        merge_adjacent=merge_adjacent
    )

    # 4) 準備檔名
    out_path = Path(out_csv_base)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # stints csv
    st_csv = out_path.with_suffix(".csv")
    print(f"[DEBUG] Writing stints to: {st_csv}")
    st.to_csv(st_csv, index=False, encoding="utf-8-sig")
    print(f"[stints] saved → {st_csv}, rows={len(st)}")

    # 5) PBP → boxscore & minutes（用 rotation 算）
    pbp_box = build_pbp_boxscore(df_e, st)
    pbp_min = compute_pbp_minutes_from_rotations(df_e)
    pbp_box_q = build_pbp_boxscore_by_period(df_e, st)

    # 6) 官方 stats → boxscore & minutes
    #    這裡一定會先賦值，再做 isinstance 檢查，就不會有 UnboundLocalError
    try:
        stats_box = flatten_stats_boxscore(stats)
    except Exception as e:
        print("[ERROR] flatten_stats_boxscore failed:", e)
        stats_box = pd.DataFrame()

    try:
        stats_min = flatten_stats_minutes(stats)
    except Exception as e:
        print("[ERROR] flatten_stats_minutes failed:", e)
        stats_min = pd.DataFrame()

    stats_box_q = flatten_stats_boxscore_by_period(stats)

    # 保險：如果哪裡沒照我們想像回傳 DataFrame，這裡強制轉
    if not isinstance(stats_box, pd.DataFrame):
        print("[WARN] stats_box is not DataFrame, force convert via pd.DataFrame(...)")
        stats_box = pd.DataFrame(stats_box)
    if not isinstance(stats_min, pd.DataFrame):
        print("[WARN] stats_min is not DataFrame, force convert via pd.DataFrame(...)")
        stats_min = pd.DataFrame(stats_min)

    # 7) 準備各種輸出目錄（按照類型分資料夾）
    base_dir = out_path.parent  # ex: outputs/season_2024
    game_id_str = str(game_id)  # 用 game_id 當檔名

    pbp_box_dir = base_dir / "pbp_box"
    stats_box_dir = base_dir / "stats_box"
    pbp_min_dir = base_dir / "pbp_minutes"
    stats_min_dir = base_dir / "stats_minutes"
    pbp_box_q_dir = base_dir / "pbp_box_by_period"
    stats_box_q_dir = base_dir / "stats_box_by_period"

    # 建立子資料夾
    for d in [pbp_box_dir, stats_box_dir, pbp_min_dir, stats_min_dir, pbp_box_q_dir, stats_box_q_dir]:
        d.mkdir(parents=True, exist_ok=True)

        # 8) 存檔：檔名 = <game_id>.csv
    pbp_box_csv = pbp_box_dir / f"{game_id_str}.csv"
    stats_box_csv = stats_box_dir / f"{game_id_str}.csv"
    pbp_min_csv = pbp_min_dir / f"{game_id_str}.csv"
    stats_min_csv = stats_min_dir / f"{game_id_str}.csv"
    pbp_box_q_csv = pbp_box_q_dir / f"{game_id_str}.csv"
    stats_box_q_csv = stats_box_q_dir / f"{game_id_str}.csv"

    if not pbp_box.empty:
        pbp_box.to_csv(pbp_box_csv, index=False, encoding="utf-8-sig")
        print(f"[boxscore] pbp_box saved → {pbp_box_csv}, rows={len(pbp_box)}")
    else:
        print("[boxscore] pbp_box is empty, skip saving")

    if not stats_box.empty:
        stats_box.to_csv(stats_box_csv, index=False, encoding="utf-8-sig")
        print(f"[boxscore] stats_box saved → {stats_box_csv}, rows={len(stats_box)}")
    else:
        print("[boxscore] stats_box is empty, skip saving")

    if not pbp_min.empty:
        pbp_min.to_csv(pbp_min_csv, index=False, encoding="utf-8-sig")
        print(f"[minutes] pbp_minutes saved → {pbp_min_csv}, rows={len(pbp_min)}")
    else:
        print("[minutes] pbp_minutes is empty, skip saving")

    if not stats_min.empty:
        stats_min.to_csv(stats_min_csv, index=False, encoding="utf-8-sig")
        print(f"[minutes] stats_minutes saved → {stats_min_csv}, rows={len(stats_min)}")
    else:
        print("[minutes] stats_minutes is empty, skip saving")

    if not pbp_box_q.empty:
        pbp_box_q.to_csv(pbp_box_q_csv, index=False, encoding="utf-8-sig")
        print(f"[boxscore] pbp_box_by_period saved → {pbp_box_q_csv}, rows={len(pbp_box_q)}")
    else:
        print("[boxscore] pbp_box_by_period is empty, skip saving")

    if not stats_box_q.empty:
        stats_box_q.to_csv(stats_box_q_csv, index=False, encoding="utf-8-sig")
        print(f"[boxscore] stats_box_by_period saved → {stats_box_q_csv}, rows={len(stats_box_q)}")
    else:
        print("[boxscore] stats_box_by_period is empty, skip saving")

        # 9) audit：總分 / boxscore / minutes / 每節
    audit_totals(st, df_e, df_s, stats)
    audit_boxscore(pbp_box, stats_box)
    audit_minutes(pbp_min, stats_min)
    audit_boxscore_by_period(pbp_box_q, stats_box_q)

        # 回傳 stints，給 batch 模式合併用
    return st


# ---------------------------
# CLI
# ---------------------------
def main():
    print("[DEBUG] tpbl_pbp2stints main() STARTED")

    ap = argparse.ArgumentParser()

    # 單場
    ap.add_argument("--game-id", type=int, help="單一比賽 ID")

    # 多場
    ap.add_argument("--game-ids", type=int, nargs="*", help="多場比賽 ID，空白分隔")

    # 區間模式（9-134 或 9-20,30-50）
    ap.add_argument("--game-id-range", type=str, help="連續 game_id 範圍，例如 9-134 或 9-20,30-35")

    ap.add_argument("--season", required=True)
    ap.add_argument("--league", default="TPBL")

    # 單場輸出
    ap.add_argument("--out-csv", help="單場 stints 輸出路徑 (ex: outputs/151_stints.csv)")

    # 批次輸出目錄
    ap.add_argument("--out-dir", help="多場時的輸出目錄 (ex: outputs/)")

    ap.add_argument("--filter-5v5", action="store_true")
    ap.add_argument("--no-merge", action="store_true")

    args = ap.parse_args()
    # game id 區間，例如 9-134

    # 判斷模式
    # 統一整理要跑哪些 game_ids
    game_ids_list = []

    # 1) 明確給 --game-ids 時
    if args.game_ids:
        game_ids_list.extend(args.game_ids)

    # 2) 用 --game-id-range 指定區間，例如 "9-134" 或 "9-20,30-35"
    if args.game_id_range:
        parts = args.game_id_range.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                if start > end:
                    start, end = end, start
                game_ids_list.extend(list(range(start, end + 1)))
            else:
                game_ids_list.append(int(part))

    # 去重排序
    game_ids_list = sorted(set(game_ids_list))

    # 判斷模式：有任何一種多場指定 → 批次模式
    if game_ids_list:
        if not args.out_dir:
            raise ValueError("使用 --game-ids 或 --game-id-range 批次模式時，必須提供 --out-dir")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_stints = []  # 用來存整季 stints

        for gid in game_ids_list:
            base = out_dir / f"{gid}_stints"
            st = process_single_game(
                game_id=gid,
                season=args.season,
                league=args.league,
                out_csv_base=str(base),
                filter_5v5=args.filter_5v5,
                merge_adjacent=not args.no_merge
            )
            if st is not None and not st.empty:
                all_stints.append(st)

        # 合併所有 stints → 一個大檔
        if all_stints:
            all_st_df = pd.concat(all_stints, ignore_index=True)
            combined_path = out_dir / f"stints_{args.league}_{args.season}_{game_ids_list[0]}-{game_ids_list[-1]}.csv"
            all_st_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
            print(f"[stints] combined season stints saved → {combined_path}, rows={len(all_st_df)}")
        else:
            print("[stints] no stints generated in batch mode, skip combined file.")

    else:
        # 單場模式
        if args.game_id is None or not args.out_csv:
            raise ValueError("單場模式必須提供 --game-id 與 --out-csv")
        process_single_game(
            game_id=args.game_id,
            season=args.season,
            league=args.league,
            out_csv_base=args.out_csv.replace(".csv", ""),
            filter_5v5=args.filter_5v5,
            merge_adjacent=not args.no_merge
        )


if __name__ == "__main__":
    main()