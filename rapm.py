# rapm.py
# 用法：
#   基本：
#     ./env/bin/python -u rapm.py \
#       --season-csv outputs/season_stints_TPBL_24-25.csv \
#       --out-csv outputs/rapm_sill_24-25.csv
#
#   指定 λ 並回報RMSE：
#     ./env/bin/python -u rapm.py \
#       --season-csv outputs/season_stints_TPBL_24-25.csv \
#       --out-csv outputs/rapm_sill_24-25.csv \
#       --lambda 8000


import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------- 解析 stints 裡的 ID & 姓名 ----------------- #

def parse_ids(val):
    """將 home_player_ids / away_player_ids 解析成 tuple[int,...]"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return tuple()
    if isinstance(val, (list, tuple, np.ndarray)):
        return tuple(int(x) for x in val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return tuple()
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return tuple(int(x) for x in obj)
        except Exception:
            s = s.strip("()[]")
            parts = [p for p in s.replace(",", " ").split() if p]
            try:
                return tuple(int(p) for p in parts)
            except Exception:
                return tuple()
    return tuple()


def parse_players_field(val):
    """
    解析 home_players / away_players:
      預期格式：[("姓名", 30), ("名字", 33), ...] 或同型態的 tuple/list
    回傳 list[(name, player_id)]
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple)):
        pairs = []
        for x in val:
            try:
                name, pid = x[0], int(x[1])
                pairs.append((str(name), pid))
            except Exception:
                continue
        return pairs
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            obj = ast.literal_eval(s)
            return parse_players_field(obj)
        except Exception:
            return []
    return []


def build_player_name_map_from_stints(df: pd.DataFrame):
    """
    從 season_stints 中的 home_players / away_players 建立 player_id -> name 對照。
    若沒有這兩欄，回傳空 dict。
    """
    pid_to_name = {}

    if "home_players" not in df.columns and "away_players" not in df.columns:
        return pid_to_name

    for col in ["home_players", "away_players"]:
        if col not in df.columns:
            continue
        for val in df[col].dropna():
            for name, pid in parse_players_field(val):
                if not name:
                    continue
                pid = int(pid)
                if pid not in pid_to_name or not pid_to_name[pid]:
                    pid_to_name[pid] = name

    return pid_to_name


# ----------------- 設計矩陣 ----------------- #

def build_design_matrix_overall(df: pd.DataFrame):
    """
    overall RAPM（net rating per 100 poss）：
      y = (home_pts - away_pts) / poss * 100
      X: 每位球員一欄 + home-court 一欄
        - home 球員：+1
        - away 球員：-1
      權重：stint 的 possessions
    回傳：
      X (N, P+1), y (N,), w (N,), player_ids, home_adv_col_index
    """
    home_ids_series = df["home_player_ids"].apply(parse_ids)
    away_ids_series = df["away_player_ids"].apply(parse_ids)

    # 收集所有球員 id
    all_ids = set()
    for ids in home_ids_series:
        all_ids.update(ids)
    for ids in away_ids_series:
        all_ids.update(ids)
    player_ids = sorted(int(pid) for pid in all_ids)

    pid_to_col = {pid: j for j, pid in enumerate(player_ids)}

    N = len(df)
    P = len(player_ids)
    X = np.zeros((N, P + 1), dtype=float)  # +1 給 home-court
    y = np.zeros(N, dtype=float)
    w = np.zeros(N, dtype=float)

    home_pts = df["home_pts"].to_numpy(dtype=float)
    away_pts = df["away_pts"].to_numpy(dtype=float)
    poss = df["possessions"].to_numpy(dtype=float)

    net = home_pts - away_pts
    poss_safe = np.where(poss <= 0, 1.0, poss)
    y[:] = (net / poss_safe) * 100.0
    w[:] = poss

    for i, (h_ids, a_ids) in enumerate(zip(home_ids_series, away_ids_series)):
        for pid in h_ids:
            j = pid_to_col.get(pid)
            if j is not None:
                X[i, j] += 1.0
        for pid in a_ids:
            j = pid_to_col.get(pid)
            if j is not None:
                X[i, j] -= 1.0
        # home-court 常數項：最後一欄
        X[i, P] = 1.0

    home_adv_col = P
    return X, y, w, player_ids, home_adv_col


# ----------------- 設計矩陣：Sill 風格 ORAPM / DRAPM ----------------- #

def build_design_matrix_off_def(df: pd.DataFrame, player_ids):
    """
      offense / defense 分開：
      E(points for offense team per 100 poss)
        = intercept + sum(off_coeff[offense players]) + sum(def_coeff[defense players])

      對每個 stint 產生 2 筆 row：
        - 一筆針對 home offense：
            y = home_pts / poss * 100
            X: home players → offense +1, away players → defense +1
        - 一筆針對 away offense：
            y = away_pts / poss * 100
            X: away players → offense +1, home players → defense +1

      防守係數越小越好（代表壓低對手得分）。
    """
    home_ids_series = df["home_player_ids"].apply(parse_ids)
    away_ids_series = df["away_player_ids"].apply(parse_ids)

    home_pts = df["home_pts"].to_numpy(dtype=float)
    away_pts = df["away_pts"].to_numpy(dtype=float)
    poss = df["possessions"].to_numpy(dtype=float)

    P = len(player_ids)
    off_index = {pid: j for j, pid in enumerate(player_ids)}              # 0..P-1
    def_index = {pid: j + P for j, pid in enumerate(player_ids)}          # P..2P-1

    N = len(df)
    rows = 2 * N
    D = 2 * P + 1  # offense P + defense P + intercept

    X = np.zeros((rows, D), dtype=float)
    y = np.zeros(rows, dtype=float)
    w = np.zeros(rows, dtype=float)

    intercept_col = 2 * P

    for i, (h_ids, a_ids, hp, ap, p) in enumerate(zip(
            home_ids_series, away_ids_series, home_pts, away_pts, poss)):
        poss_safe = p if p > 0 else 1.0

        # row for home offense
        r = 2 * i
        y[r] = 100.0 * hp / poss_safe
        w[r] = p
        for pid in h_ids:
            j = off_index.get(pid)
            if j is not None:
                X[r, j] += 1.0
        for pid in a_ids:
            j = def_index.get(pid)
            if j is not None:
                X[r, j] += 1.0
        X[r, intercept_col] = 1.0

        # row for away offense
        r2 = r + 1
        y[r2] = 100.0 * ap / poss_safe
        w[r2] = p
        for pid in a_ids:
            j = off_index.get(pid)
            if j is not None:
                X[r2, j] += 1.0
        for pid in h_ids:
            j = def_index.get(pid)
            if j is not None:
                X[r2, j] += 1.0
        X[r2, intercept_col] = 1.0

    return X, y, w, intercept_col


# ----------------- 一般化 Ridge + CV ----------------- #

def fit_ridge(X, y, w, lam, unpenalized_cols=None):
    """
    Weighted ridge:
      minimize sum_i w_i (y_i - x_i·β)^2 + λ * sum_j β_j^2 (j not in unpenalized_cols)
    """
    if unpenalized_cols is None:
        unpenalized_cols = set()
    else:
        unpenalized_cols = set(unpenalized_cols)

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    P_plus = X.shape[1]
    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw

    reg = np.zeros((P_plus, P_plus), dtype=float)
    for j in range(P_plus):
        reg[j, j] = 0.0 if j in unpenalized_cols else lam

    beta = np.linalg.solve(XtX + reg, Xty)
    return beta


def cross_val_rmse(X, y, w, lam, unpenalized_cols=None, k_folds=5, random_state=42):
    """
    對 RAPM 矩陣做 K-fold CV。
    回傳該 λ 的 RMSE。
    """
    if unpenalized_cols is None:
        unpenalized_cols = set()
    else:
        unpenalized_cols = set(unpenalized_cols)

    rng = np.random.default_rng(random_state)
    N = len(y)
    indices = np.arange(N)
    rng.shuffle(indices)

    folds = np.array_split(indices, k_folds)
    sq_errors = []
    counts = []

    for k in range(k_folds):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != k])

        X_tr, y_tr, w_tr = X[train_idx], y[train_idx], w[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        beta = fit_ridge(X_tr, y_tr, w_tr, lam, unpenalized_cols)
        y_pred = X_va @ beta

        err = (y_va - y_pred) ** 2
        sq_errors.append(err.sum())
        counts.append(len(y_va))

    mse = np.sum(sq_errors) / np.sum(counts)
    rmse = float(np.sqrt(mse))
    return rmse


# ----------------- 主流程 ----------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--lambda", dest="lam", type=float,
                    help="若指定，使用此 λ 並回報該 λ 的 CV RMSE；未指定則跑預設 λ grid")
    ap.add_argument("--k-folds", type=int, default=5)
    args = ap.parse_args()

    season_path = Path(args.season_csv)
    df = pd.read_csv(season_path)

    # player_id -> name mapping
    pid_to_name = build_player_name_map_from_stints(df)

    # RAPM 矩陣
    X_overall, y_overall, w_overall, player_ids, home_adv_col = build_design_matrix_overall(df)

    # 1) Cross-validation for λ
    if args.lam is not None:
        lam_grid = [args.lam]
        print(f"[cv] user-specified λ = {args.lam}")
    else:
        lam_grid = [125.0, 250.0, 400.0, 600.0, 800.0, 1000.0,
                    1500.0, 2000.0, 3000.0, 5000.0, 8000.0, 12000.0]

    cv_results = []
    for lam in lam_grid:
        rmse = cross_val_rmse(
            X_overall, y_overall, w_overall,
            lam, unpenalized_cols={home_adv_col},
            k_folds=args.k_folds
        )
        cv_results.append((lam, rmse))
        print(f"[cv] λ={lam:8.1f} → RMSE={rmse:8.3f}")

    if args.lam is not None:
        lam_use = float(args.lam)
        rmse_use = cv_results[0][1]
        print(f"[cv] using user λ = {lam_use} (RMSE={rmse_use:.3f})")
    else:
        rmse_array = np.array([r[1] for r in cv_results])
        lam_array = np.array([r[0] for r in cv_results])
        idx_min = int(rmse_array.argmin())
        lam_min = float(lam_array[idx_min])
        rmse_min = float(rmse_array[idx_min])

        rmse_std = float(rmse_array.std(ddof=1)) if len(rmse_array) > 1 else 0.0
        threshold = rmse_min + rmse_std

        lam_1se = lam_min
        for l, r in sorted(zip(lam_array, rmse_array), key=lambda x: x[0]):
            if r <= threshold:
                lam_1se = float(l)
                break

        lam_use = lam_1se
        print(f"[cv] λ_minRMSE = {lam_min:.1f} (RMSE={rmse_min:.3f})")
        print(f"[cv] λ_1SE     = {lam_1se:.1f} (threshold={threshold:.3f})")
        print(f"[done] λ selected for all models (RAPM / ORAPM / DRAPM): λ={lam_use:.1f}")

    # 2) 以 lam_use 擬合 RAPM
    beta_overall = fit_ridge(
        X_overall, y_overall, w_overall,
        lam_use, unpenalized_cols={home_adv_col}
    )
    P = len(player_ids)
    player_coefs_overall = beta_overall[:P]
    home_court_adv = beta_overall[home_adv_col]
    print(f"[info] overall home-court advantage (pts/100): {home_court_adv:.3f}")

    # 3) 以同一 λ 擬合 ORAPM / DRAPM
    X_offdef, y_offdef, w_offdef, offdef_intercept_col = build_design_matrix_off_def(df, player_ids)
    beta_offdef = fit_ridge(
        X_offdef, y_offdef, w_offdef,
        lam_use, unpenalized_cols={offdef_intercept_col}
    )

    # 拆出 offensive / defensive 係數
    orapm_coefs = beta_offdef[0:P]
    drapm_coefs = beta_offdef[P:2 * P]
    # intercept = beta_offdef[offdef_intercept_col]  # 如有需要可以印出 baseline ORtg

    # 4) 計算 on_court_poss 與 games_played & poss_per_game
    poss = df["possessions"].to_numpy(dtype=float)
    home_ids_series = df["home_player_ids"].apply(parse_ids)
    away_ids_series = df["away_player_ids"].apply(parse_ids)

    on_court_poss = {pid: 0.0 for pid in player_ids}
    games_played = {pid: set() for pid in player_ids}

    game_ids = df["game_id"].tolist()

    for game_id, st_poss, h_ids, a_ids in zip(game_ids, poss, home_ids_series, away_ids_series):
        for pid in h_ids:
            on_court_poss[pid] += st_poss
            games_played[pid].add(game_id)
        for pid in a_ids:
            on_court_poss[pid] += st_poss
            games_played[pid].add(game_id)

    # 5) 組合輸出表：四捨五入到小數點第二位 & 場均 poss
    rows = []
    for idx, pid in enumerate(player_ids):
        rapm = float(player_coefs_overall[idx])
        orapm = float(orapm_coefs[idx])
        drapm = float(drapm_coefs[idx])  # 數值越小越好（代表壓低對手得分）

        total_poss = float(on_court_poss.get(pid, 0.0))
        gp = len(games_played.get(pid, set()))
        poss_per_game = total_poss / gp if gp > 0 else 0.0

        rows.append({
            "player_id": int(pid),
            "player_name": pid_to_name.get(pid, ""),
            "rapm_per100": round(rapm, 2),
            "orapm_per100": round(orapm, 2),
            "drapm_per100": round(drapm, 2),  # smaller is better
            "on_court_poss": round(total_poss, 1),
            "games_played": int(gp),
            "poss_per_game": round(poss_per_game, 1),
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("rapm_per100", ascending=False).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    cv_path = out_path.with_name(out_path.stem + "_cv_results.csv")
    cv_df = pd.DataFrame(cv_results, columns=["lambda", "rmse"])
    cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")

    print(f"[save] players → {out_path}")
    print(f"[save] CV      → {cv_path}")

    print("\nTop 10 players by RAPM / ORAPM / DRAPM (pts/100 poss):")
    print(out_df[["player_id", "player_name",
                  "rapm_per100", "orapm_per100", "drapm_per100",
                  "on_court_poss", "games_played", "poss_per_game"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()