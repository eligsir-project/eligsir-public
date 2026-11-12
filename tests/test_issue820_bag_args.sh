#!/usr/bin/env bash
# Regression test for issue #820 (bag-job TRAIN_EXTRA_ARGS quoting) and the
# follow-up review on PR #835 that found the first fix broken.
#
# NON-INTERACTIVE, no GPU/ROS2/Docker required.
#
# The first version of this test exercised the tokenization logic copy-pasted
# in isolation and passed while the real script was broken on every
# invocation (unbound-variable crash, or the script's own --model-path/
# --idle-timeout silently destroyed). This version drives the SCRIPT'S OWN
# argument-parsing code, extracted verbatim from the live file at test time
# (never a hand-maintained copy), through its real entry path: BAG_PATH
# positional consumption -> TRAIN_EXTRA_ARGS tokenization -> --gt-pose-topic
# sentinel extraction -> the script's own --model-path/--idle-timeout/etc CLI
# parser, in that order, exactly as 05_run_offline_rosbag.sh itself runs them.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'  # No Color

test_count=0
pass_count=0

log_test() {
    # NOTE: `((test_count++))` (a bare arithmetic command) returns the
    # PRE-increment value as its exit status; when that value is 0 (the very
    # first call) the command's exit status is itself 0-is-success... but
    # bash's arithmetic-command truthiness is the opposite of that for `-e`
    # purposes: `(( 0 ))` reports failure (exit 1), which `set -e` (in force
    # in this file) then treats as the whole script failing on the FIRST
    # test registration, before it ever prints anything. Use plain
    # arithmetic substitution instead, which has no such command-exit-status
    # side effect.
    test_count=$((test_count + 1))
    echo -n "Test $test_count: $1 ... "
}

log_pass() {
    pass_count=$((pass_count + 1))
    echo -e "${GREEN}PASS${NC}"
}

log_skip() {
    # Environment-gated check (e.g. shellcheck absent) -- neither a pass nor
    # a fail of this repo's own code, so it must not affect exit status.
    pass_count=$((pass_count + 1))
    echo "SKIP: $1"
}

log_fail() {
    echo -e "${RED}FAIL${NC}: $1"
}

SCRIPT_PATH="human_launch_scripts/05_run_offline_rosbag.sh"

# Extract the script's real argument-parsing prologue (from the top through
# the end of its own --model-path/--idle-timeout CLI parser, and NOT one
# line further -- everything after that point does real podman/ros2/docker
# work this test must not attempt). Located by anchoring on the literal
# `export ROS_DOMAIN_ID=` line that immediately follows the parser in the
# real file, so this stays correct even if line numbers shift.
PARSE_END_LINE="$(grep -n '^export ROS_DOMAIN_ID=' "$SCRIPT_PATH" | head -1 | cut -d: -f1)"
if [ -z "$PARSE_END_LINE" ]; then
    echo "FATAL: could not locate the end of $SCRIPT_PATH's argument-parsing block (export ROS_DOMAIN_ID= line missing/moved). Update this test's anchor." >&2
    exit 1
fi
PARSE_END_LINE=$((PARSE_END_LINE - 1))

HARNESS_DIR="$(mktemp -d)"
trap 'rm -rf "$HARNESS_DIR"' EXIT
HARNESS_FILE="$HARNESS_DIR/parse_prologue.sh"
sed -n "1,${PARSE_END_LINE}p" "$SCRIPT_PATH" > "$HARNESS_FILE"

# run_prologue <TRAIN_EXTRA_ARGS value> -- <bag path> <script's own CLI args...>
# Sources the extracted prologue in a subshell (so each test starts clean and
# a real `exit 1` from the script's own "Unknown parameter" branch or set -u
# failure is caught as a subshell failure, not a test-process kill) and
# prints out the resulting MODEL_PATH / IDLE_TIMEOUT / POSE_TOPIC /
# USE_BAKED_POSES / _parsed_extra_args so the caller can assert on them.
# The FIRST positional argument after `--` must be a (fake) bag path --
# real invocations of 05_run_offline_rosbag.sh always take $1 as BAG_PATH
# before parsing any flags, and the extracted prologue does exactly that
# unconditionally (`BAG_PATH="$1"; shift`), so a test omitting it would be
# testing a call shape the real script never accepts.
run_prologue() {
    local extra_args_val="$1"
    shift
    if [ "${1:-}" = "--" ]; then shift; fi
    (
        if [ "$extra_args_val" = "__UNSET__" ]; then
            unset TRAIN_EXTRA_ARGS 2>/dev/null || true
        else
            export TRAIN_EXTRA_ARGS="$extra_args_val"
        fi
        set -euo pipefail
        # shellcheck source=/dev/null
        source "$HARNESS_FILE" "$@"
        echo "__RESULT_MODEL_PATH__=$MODEL_PATH"
        echo "__RESULT_IDLE_TIMEOUT__=$IDLE_TIMEOUT"
        echo "__RESULT_POSE_TOPIC__=$POSE_TOPIC"
        echo "__RESULT_USE_BAKED_POSES__=$USE_BAKED_POSES"
        echo "__RESULT_NPARSED__=${#_parsed_extra_args[@]}"
        local _i=0
        for _a in "${_parsed_extra_args[@]:-}"; do
            echo "__RESULT_PARSED_${_i}__=$_a"
            _i=$((_i + 1))
        done
    )
}

get_field() {
    # get_field <output> <field-name>
    printf '%s\n' "$1" | grep "^__RESULT_${2}__=" | head -1 | cut -d= -f2-
}

# ---------------------------------------------------------------------------
# Test 1: real entry path, own flags + multi-word TRAIN_EXTRA_ARGS together.
# This is exactly Blocker 2 from the review: the script's own --model-path
# and --idle-timeout must survive being parsed AFTER a multi-word
# TRAIN_EXTRA_ARGS has been tokenized, and the multi-word purpose must come
# through as ONE array element, not word-split.
# ---------------------------------------------------------------------------
log_test "own CLI flags survive + multi-word TRAIN_EXTRA_ARGS is one element (real entry path)"
out="$(run_prologue \
    '--experiment_purpose "a b c" --foo bar' \
    -- /fake/bag --model-path /out/x --idle-timeout 900)"
model_path="$(get_field "$out" MODEL_PATH)"
idle_timeout="$(get_field "$out" IDLE_TIMEOUT)"
nparsed="$(get_field "$out" NPARSED)"
p0="$(get_field "$out" PARSED_0)"
p1="$(get_field "$out" PARSED_1)"
p2="$(get_field "$out" PARSED_2)"
p3="$(get_field "$out" PARSED_3)"
if [ "$model_path" = "/out/x" ] && [ "$idle_timeout" = "900" ] && \
   [ "$nparsed" = "4" ] && [ "$p0" = "--experiment_purpose" ] && \
   [ "$p1" = "a b c" ] && [ "$p2" = "--foo" ] && [ "$p3" = "bar" ]; then
    log_pass
else
    log_fail "MODEL_PATH=$model_path IDLE_TIMEOUT=$idle_timeout NPARSED=$nparsed p0=[$p0] p1=[$p1] p2=[$p2] p3=[$p3]"
fi

# ---------------------------------------------------------------------------
# Test 2: TRAIN_EXTRA_ARGS unset entirely (every manual invocation, every
# ablator job with no extra_args). Must not die under set -u (Blocker 1).
# ---------------------------------------------------------------------------
log_test "TRAIN_EXTRA_ARGS unset does not crash the real entry path"
if out="$(run_prologue '__UNSET__' -- /fake/bag --model-path /out/y --idle-timeout 30)"; then
    model_path="$(get_field "$out" MODEL_PATH)"
    idle_timeout="$(get_field "$out" IDLE_TIMEOUT)"
    nparsed="$(get_field "$out" NPARSED)"
    if [ "$model_path" = "/out/y" ] && [ "$idle_timeout" = "30" ] && [ "$nparsed" = "0" ]; then
        log_pass
    else
        log_fail "ran without crashing but wrong state: MODEL_PATH=$model_path IDLE_TIMEOUT=$idle_timeout NPARSED=$nparsed"
    fi
else
    log_fail "script's own prologue exited nonzero on unset TRAIN_EXTRA_ARGS (this is Blocker 1 from the review)"
fi

# ---------------------------------------------------------------------------
# Test 3: TRAIN_EXTRA_ARGS empty string (also common: ablator sets the env
# var but the job's extra_args is "").
# ---------------------------------------------------------------------------
log_test "TRAIN_EXTRA_ARGS empty string does not crash the real entry path"
if out="$(run_prologue '' -- /fake/bag --model-path /out/z --idle-timeout 45)"; then
    model_path="$(get_field "$out" MODEL_PATH)"
    idle_timeout="$(get_field "$out" IDLE_TIMEOUT)"
    nparsed="$(get_field "$out" NPARSED)"
    if [ "$model_path" = "/out/z" ] && [ "$idle_timeout" = "45" ] && [ "$nparsed" = "0" ]; then
        log_pass
    else
        log_fail "ran without crashing but wrong state: MODEL_PATH=$model_path IDLE_TIMEOUT=$idle_timeout NPARSED=$nparsed"
    fi
else
    log_fail "script's own prologue exited nonzero on empty TRAIN_EXTRA_ARGS"
fi

# ---------------------------------------------------------------------------
# Test 4: --gt-pose-topic sentinel path, real entry path, WITH a multi-word
# purpose also present (own flags + sentinel stripping + tokenization all at
# once, exactly as a real queued job would combine them).
# ---------------------------------------------------------------------------
log_test "--gt-pose-topic sentinel + multi-word purpose, real entry path"
out="$(run_prologue \
    '--gt-pose-topic /gt/colmap_pose --experiment_purpose "a b c" --other flag' \
    -- /fake/bag --model-path /out/gt --idle-timeout 60)"
model_path="$(get_field "$out" MODEL_PATH)"
idle_timeout="$(get_field "$out" IDLE_TIMEOUT)"
pose_topic="$(get_field "$out" POSE_TOPIC)"
use_baked="$(get_field "$out" USE_BAKED_POSES)"
nparsed="$(get_field "$out" NPARSED)"
p0="$(get_field "$out" PARSED_0)"
p1="$(get_field "$out" PARSED_1)"
p2="$(get_field "$out" PARSED_2)"
p3="$(get_field "$out" PARSED_3)"
if [ "$model_path" = "/out/gt" ] && [ "$idle_timeout" = "60" ] && \
   [ "$pose_topic" = "/gt/colmap_pose" ] && [ "$use_baked" = "1" ] && \
   [ "$nparsed" = "4" ] && [ "$p0" = "--experiment_purpose" ] && \
   [ "$p1" = "a b c" ] && [ "$p2" = "--other" ] && [ "$p3" = "flag" ]; then
    log_pass
else
    log_fail "MODEL_PATH=$model_path IDLE_TIMEOUT=$idle_timeout POSE_TOPIC=$pose_topic USE_BAKED_POSES=$use_baked NPARSED=$nparsed p0=[$p0] p1=[$p1] p2=[$p2] p3=[$p3]"
fi

# ---------------------------------------------------------------------------
# Test 5: tokenizer does not execute embedded command substitution (the
# eval-vs-xargs tradeoff called out in the code comment / PR description).
# ---------------------------------------------------------------------------
log_test "TRAIN_EXTRA_ARGS with embedded \$(...) is treated as inert literal text"
CANARY="$HARNESS_DIR/PWNED"
rm -f "$CANARY"
out="$(run_prologue \
    "--experiment_purpose \"\$(touch $CANARY)\" --foo bar" \
    -- /fake/bag --model-path /out/inj --idle-timeout 5)"
p1="$(get_field "$out" PARSED_1)"
if [ ! -e "$CANARY" ] && [ "$p1" = "\$(touch $CANARY)" ]; then
    log_pass
else
    log_fail "canary exists=$( [ -e "$CANARY" ] && echo yes || echo no ), p1=[$p1] -- command substitution was executed instead of staying literal"
fi

# ---------------------------------------------------------------------------
# Test 6: malformed/unterminated quoting in TRAIN_EXTRA_ARGS must fail LOUDLY
# (nonzero exit + a message naming TRAIN_EXTRA_ARGS), not silently truncate
# the tokenized array and continue. Found in a second round of review: xargs
# runs on the read side of a `< <(...)` process substitution, whose exit
# status `pipefail` does not observe, so the first cut of this fix let a bad
# quote silently drop every trailing argument instead of erroring -- exactly
# the silent-misconfiguration class issue #820 exists to eliminate (a job
# training with a shorter, different argv than its extra_args says, with no
# visible signal).
# ---------------------------------------------------------------------------
log_test "malformed/unterminated quote in TRAIN_EXTRA_ARGS fails loudly (real entry path)"
malformed_exit=0
malformed_output="$(run_prologue \
    '--experiment_purpose "unterminated --other flag' \
    -- /fake/bag --model-path /out/malformed --idle-timeout 5 2>&1)" || malformed_exit=$?
if [ "$malformed_exit" -ne 0 ] && printf '%s' "$malformed_output" | grep -q "TRAIN_EXTRA_ARGS"; then
    log_pass
else
    log_fail "expected nonzero exit + a TRAIN_EXTRA_ARGS-naming error, got exit=$malformed_exit output=[$malformed_output]"
fi

# ---------------------------------------------------------------------------
# Test 7: bash -n syntax validation of the real script.
# ---------------------------------------------------------------------------
log_test "05_run_offline_rosbag.sh syntax validation (bash -n)"
if bash -n "$SCRIPT_PATH" 2>/dev/null; then
    log_pass
else
    log_fail "Script has syntax errors"
fi

# ---------------------------------------------------------------------------
# Test 8: shellcheck, if available (informational skip otherwise -- do not
# fail the suite in an environment without it installed).
# ---------------------------------------------------------------------------
log_test "shellcheck on 05_run_offline_rosbag.sh (skipped if shellcheck absent)"
if command -v shellcheck >/dev/null 2>&1; then
    if shellcheck "$SCRIPT_PATH"; then
        log_pass
    else
        log_fail "shellcheck reported issues"
    fi
else
    log_skip "shellcheck not installed"
fi

# ---------------------------------------------------------------------------
# Test 9: timeout bound on the bag-playback-complete publisher was raised
# from the unsafe `timeout 10` (issue #820 review's non-blocking note) and
# is still a hard, finite bound (not removed/unbounded).
# ---------------------------------------------------------------------------
log_test "bag-playback-complete publisher timeout is bounded and > 10s"
timeout_val="$(grep -oE '^timeout [0-9]+ "\$RUNTIME" run' "$SCRIPT_PATH" | head -1 | awk '{print $2}')"
if [ -n "$timeout_val" ] && [ "$timeout_val" -gt 10 ] && [ "$timeout_val" -le 300 ]; then
    log_pass
else
    log_fail "expected a bounded timeout in (10, 300]s guarding the publisher, got [$timeout_val]"
fi

# ---------------------------------------------------------------------------
# Tests 10-14: issue #858. PR #835 deleted the scalar that carried the
# caller-supplied train.py args, but left one consumer of it behind in the
# ORB-SLAM3 pose branch. That crashed under `set -u` (defect 1) -- and the
# naive `:-` fix would have made the script run while SILENTLY dropping
# --ros2_live_pose_camera_frame_convention from the training argv (defect 2),
# i.e. every ORB-SLAM3 bag/live run reconstructing with the wrong pose
# rotation convention. Defect 2 is the one that must stay guarded: it is not
# self-announcing.
# ---------------------------------------------------------------------------

# Test 10: the argv assertion PASSES when the flag is present.
log_test "#858: _assert_train_argv_has accepts an argv containing the pose-convention flag"
if out="$(
    set -euo pipefail
    # shellcheck source=/dev/null
    source "$HARNESS_FILE" /fake/bag --model-path /out/a858 --idle-timeout 5
    argv=(--ros2_live_idle_timeout 30 --ros2_live_bag_replay --ros2_live_pose_camera_frame_convention link)
    _assert_train_argv_has "--ros2_live_pose_camera_frame_convention" "${argv[@]}"
    echo OK
    )"; then
    [ "$out" = "OK" ] && log_pass || log_fail "unexpected output [$out]"
else
    log_fail "assertion rejected an argv that DOES contain the flag"
fi

# Test 11: the argv assertion FAILS LOUDLY when the flag is absent. This is
# the test that actually catches defect 2 (a dropped flag) rather than the
# crash: it must exit nonzero AND name the missing flag.
log_test "#858: _assert_train_argv_has rejects an argv missing the pose-convention flag, naming it"
missing_exit=0
missing_out="$(
    set -euo pipefail
    # shellcheck source=/dev/null
    source "$HARNESS_FILE" /fake/bag --model-path /out/b858 --idle-timeout 5
    argv=(--ros2_live_idle_timeout 30 --ros2_live_bag_replay)
    _assert_train_argv_has "--ros2_live_pose_camera_frame_convention" "${argv[@]}" 2>&1
    )" || missing_exit=$?
if [ "$missing_exit" -ne 0 ] && printf '%s' "$missing_out" | grep -q -- "--ros2_live_pose_camera_frame_convention"; then
    log_pass
else
    log_fail "expected nonzero exit + a message naming the flag, got exit=$missing_exit output=[$missing_out]"
fi

# Test 12: the dead scalar is gone entirely. Case-sensitive on the uppercase
# token so the live lowercase array (_extra_train_args) does not match, and
# `TRAIN_EXTRA_ARGS` (the INPUT env var, legitimately still read) is excluded.
log_test "#858: no surviving reference to the dead uppercase scalar in the script"
orphan_hits="$(grep -o 'EXTRA_TRAIN_ARGS' "$SCRIPT_PATH" | grep -c '^EXTRA_TRAIN_ARGS$' || true)"
if [ "$orphan_hits" = "0" ]; then
    log_pass
else
    log_fail "found $orphan_hits reference(s) to the dead EXTRA_TRAIN_ARGS scalar (PR #835 orphan, issue #858)"
fi

# Test 13: the ORB-SLAM3 pose branch appends the flag to the LIVE array (the
# one the trainer command actually expands), not to some other variable.
log_test "#858: ORB-SLAM3 branch appends the pose convention to the live _extra_train_args array"
if grep -q '^\s*_extra_train_args+=("--ros2_live_pose_camera_frame_convention" "link")$' "$SCRIPT_PATH" && \
   grep -q '^\s*"\${_extra_train_args\[@\]}"$' "$SCRIPT_PATH"; then
    log_pass
else
    log_fail "expected an _extra_train_args+=(...) append AND a \"\${_extra_train_args[@]}\" expansion in the trainer command"
fi

# Test 14: the pre-launch guard is wired, and gated on the ORB-SLAM3 path only
# (the baked-pose branch must NOT require the link convention).
log_test "#858: pre-launch guard is wired and gated on USE_BAKED_POSES -eq 0"
guard_line="$(grep -n '_assert_train_argv_has "--ros2_live_pose_camera_frame_convention"' "$SCRIPT_PATH" | tail -1 | cut -d: -f1)"
if [ -n "$guard_line" ]; then
    gate_line="$(grep -n 'if \[ "\$USE_BAKED_POSES" -eq 0 \]; then' "$SCRIPT_PATH" | awk -F: -v g="$guard_line" '$1 < g {l=$1} END {print l}')"
    launch_line="$(grep -n 'ros2 run eligsir_ros live_training' "$SCRIPT_PATH" | tail -1 | cut -d: -f1)"
    if [ -n "$gate_line" ] && [ -n "$launch_line" ] && [ "$guard_line" -lt "$launch_line" ]; then
        log_pass
    else
        log_fail "guard at line [$guard_line] not gated (gate=[$gate_line]) or not before the trainer launch (launch=[$launch_line])"
    fi
else
    log_fail "no _assert_train_argv_has call for the pose-convention flag found before the trainer launch"
fi

echo ""
echo "======================================"
echo "Test Results: $pass_count / $test_count PASSED"
echo "======================================"

if [ $pass_count -eq $test_count ]; then
    exit 0
else
    exit 1
fi
