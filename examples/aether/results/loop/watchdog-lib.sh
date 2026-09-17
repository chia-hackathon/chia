#!/bin/bash
# 2026-09-14: 共用的「何時該重啟」判斷，修掉 Round 7 空轉 52 次的 bug。
#
# 背景：loop 撞到 Claude 用量上限時，log 裡可能同時出現兩種 resets 訊息：
#   session 上限 "resets Sep 14, 11pm (America/Los_Angeles)"      -> 數小時後
#   週用量上限  "(resets 2026-09-15T06:00:00+00:00)"              -> 數天後
# 舊版用 `grep ... | tail -1` 只看最後一行，挑到近的那個，於是每 40 分鐘重啟一次、
# 連續 52 次全部立刻 429，0 進展。改成取「所有 resets 中最晚的時間」，並在等待
# 超過 6 小時（等同週限）時直接放棄重啟、留下紀錄交給操作者。
#
# reset_wait_seconds <logfile>
#   印出應等待的秒數；印 -1 表示週用量上限，不要重啟。

_resets_epoch() {   # 單行 -> epoch（失敗印空字串）
  local line=$1 t tz iso target now
  now=$(date +%s)
  # Shape 1: "resets <time> (<Region/City>)"
  if [[ "$line" =~ resets[[:space:]]+(.+)\(([A-Za-z_]+(/[A-Za-z_]+)+)\) ]]; then
    t="${BASH_REMATCH[1]}"; tz="${BASH_REMATCH[2]}"
    t="${t#"${t%%[![:space:]]*}"}"; t="${t%"${t##*[![:space:]]}"}"
    t="${t%,}"
    target=$(TZ="$tz" date -d "$t" +%s 2>/dev/null)
    if [[ -n "$target" ]]; then
      (( target <= now )) && target=$(( target + 86400 ))
      echo "$target"; return
    fi
  fi
  # Shape 2: ISO 8601 帶時區
  iso=$(echo "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(Z|[+-][0-9]{2}:[0-9]{2})' | head -1)
  if [[ -n "$iso" ]]; then
    target=$(date -d "$iso" +%s 2>/dev/null)
    [[ -n "$target" ]] && echo "$target"
  fi
}

reset_wait_seconds() {
  local log=$1 now best=0 cand ct
  now=$(date +%s)
  [[ -f "$log" ]] || { echo 1800; return; }
  while IFS= read -r cand; do
    ct=$(_resets_epoch "$cand")
    [[ -n "$ct" ]] && (( ct > best )) && best=$ct
  done < <(grep -oiE 'resets[[:space:]]+[^"]*' "$log")
  if (( best > now )); then
    local w=$(( best - now + 60 ))
    (( w > 21600 )) && { echo -1; return; }   # >6h = 週用量上限
    echo "$w"; return
  fi
  echo 1800
}

# 2026-09-15: 判斷 log 是否真的撞到用量上限。
# 舊版用 grep -E 'session limit|rate limit|429' 會被 raylet 的磁碟警告誤觸
# （那些行的 pid 剛好含 429），導致 round8 跑完 4/4 後還排了一次重啟。
# 只認 loop 自己印的 "Rate limit detected on <hash> (resets ...)" 與 session limit。
is_rate_limited() {
  local log=$1
  [[ -f "$log" ]] || return 1
  grep -qiE 'Rate limit detected on|usage limit reached|session limit .*reset' "$log"
}
