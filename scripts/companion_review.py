"""P5-c blinded human-review material; never generates or simulates human scores."""

import argparse
import hashlib
import json
import secrets
from collections import Counter
from pathlib import Path

from hmph.model_gateway import FALLBACK

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/live/p4c-mention-gating-v1/report.json"
DIMENSIONS = ("naturalness", "persona", "timing", "continuity", "repair")

PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Hmph 独立盲评</title><style>
body{font:16px/1.65 system-ui;background:#f5f4f1;color:#202521;margin:0}
main{max-width:1280px;margin:auto;padding:24px}h1{margin:0 0 8px}fieldset,article,details{background:white;border:1px solid #d6ddd7;border-radius:10px;padding:16px;margin:14px 0}
input,select,button,textarea{font:inherit;padding:7px;border:1px solid #a8b6ae;border-radius:6px}button{cursor:pointer;background:#234f41;color:white}textarea{width:96%;min-height:65px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.reply{white-space:pre-wrap;border-left:4px solid #487462;padding:14px;background:#f0f5f1}.history{white-space:pre-wrap;max-height:480px;overflow:auto;font-size:14px}.fields label{display:flex;justify-content:space-between;gap:8px;margin:8px 0}.muted{color:#53665b}.nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap}@media(max-width:800px){.columns{grid-template-columns:1fr}}
</style><main><h1>Hmph 独立盲评</h1>
<p>请独立完成，不与另一位评审讨论或查看对方答案。左右组别已隐藏；评估实际发布的回复，包含拦截回复，不猜实验组。允许平局和无法判断。</p>
<p class="muted">30 组来自同一条连续轨迹，提供截至当轮的完整上下文。这次对照改变记忆提及策略；也请检查共同计划和纠错行为。请勿把片段阅读填写为真实试用。</p>
<fieldset><legend>评审与真实试用记录</legend>
<label>匿名编号 <input id="rater" placeholder="例如 R1 或 R2" autocomplete="off"></label>
<label><input id="human" type="checkbox">我是独立真人，以下评分由我本人填写</label><br>
<label>是否真实使用过 Hmph <select id="tried"><option value="no">尚未真实试用</option><option value="yes">已真实试用</option></select></label>
<label>开始 <input id="trial-start" type="date"></label><label>结束 <input id="trial-end" type="date"></label>
<p>实际使用跨度、会话情况或退出意见（未试用可留空；阅读片段的时间不算真实试用跨度）：</p><textarea id="trial-notes"></textarea>
</fieldset><div class="nav"><button id="prev">上一组</button><select id="cases"></select><button id="next">下一组</button><strong id="progress"></strong></div>
<h2 id="case-title"></h2><p id="clock" class="muted"></p><article><strong>本轮用户</strong><div id="question" class="reply"></div></article>
<div class="columns"><article><h2>回复 A</h2><div id="replyA" class="reply"></div><details><summary>展开完整前文 A</summary><div id="contextA" class="history"></div></details><div id="fieldsA" class="fields"></div></article>
<article><h2>回复 B</h2><div id="replyB" class="reply"></div><details><summary>展开完整前文 B</summary><div id="contextB" class="history"></div></details><div id="fieldsB" class="fields"></div></article></div>
<fieldset><legend>成对判断</legend><label>总体偏好 <select id="preference"><option value="">请选择</option><option value="A">A 更好</option><option value="B">B 更好</option><option value="tie">平局</option><option value="skip">无法判断／跳过</option></select></label>
<p>具体依据或问题原句（发现事实／边界问题时请填写）：</p><textarea id="comment"></textarea></fieldset>
<div class="nav"><button id="download">导出本人评分／草稿</button><label>恢复本人草稿 <input id="restore" type="file" accept="application/json"></label></div><p id="notice" class="muted">尚无评分。评分保留在当前页面；关闭前请导出草稿。不要把评分交给模型代填。</p>
<script id="pack" type="application/json">__PACK__</script><script>
const pack=JSON.parse(document.getElementById('pack').textContent),answers={},started=new Date().toISOString();let current=0;
const labels={naturalness:'自然度',persona:'角色一致',timing:'记忆提及是否合时宜',continuity:'连续性／承接',repair:'纠错后是否持续采用新值'};
const $=id=>document.getElementById(id);let loading=false;
for(let i=0;i<pack.pairs.length;i++){const o=new Option(`${i+1} / ${pack.pairs.length}`,i);$('cases').add(o)}
function selectField(side,key,label,choices){const line=document.createElement('label');line.textContent=label;const select=document.createElement('select');select.id=side+'-'+key;select.add(new Option('请选择',''));for(const [value,text] of choices)select.add(new Option(text,value));select.onchange=save;line.append(select);$('fields'+side).append(line)}
for(const side of ['A','B']){for(const [key,label] of Object.entries(labels)){const opts=[1,2,3,4,5].map(v=>[String(v),`${v} 分`]);if(key==='repair')opts.push(['na','本片段不适用']);selectField(side,key,label,opts)}for(const [key,label] of [['facts','事实正确性'],['boundaries','是否违背明确要求（称呼／禁提／暂停）']])selectField(side,key,label,[['none','无明显问题'],['problem','有具体问题'],['uncertain','无法判断']])}
function save(){if(loading)return;const p=pack.pairs[current],a={preference:$('preference').value,comment:$('comment').value,A:{},B:{}};for(const side of ['A','B'])for(const key of [...Object.keys(labels),'facts','boundaries'])a[side][key]=$(side+'-'+key).value;answers[p.id]=a;progress()}
function valid(p,a){if(!a||!['A','B','tie'].includes(a.preference))return false;for(const side of ['A','B']){for(const key of Object.keys(labels)){const v=a[side][key];if(!['1','2','3','4','5'].includes(v)&&!(key==='repair'&&v==='na'))return false}if(!['none','problem','uncertain'].includes(a[side].facts)||!['none','problem','uncertain'].includes(a[side].boundaries))return false;if((a[side].facts==='problem'||a[side].boundaries==='problem')&&!a.comment.trim())return false}return true}
function progress(){$('progress').textContent=`有效填写 ${pack.pairs.filter(p=>valid(p,answers[p.id])).length} / ${pack.pairs.length}`}
function show(index){save();current=Math.max(0,Math.min(pack.pairs.length-1,index));loading=true;const p=pack.pairs[current],a=answers[p.id]||{A:{},B:{}};$('cases').value=current;$('case-title').textContent=`第 ${current+1} 组 · ${{C1:'提及与节奏',C2:'共同线索',C3:'称呼与纠错'}[p.family]}`;$('clock').textContent=`记录时间：${p.recorded_at||'见前文'}`;$('question').textContent=p.user;for(const side of ['A','B']){$('reply'+side).textContent=p[side].reply;$('context'+side).textContent=p[side].context;for(const key of [...Object.keys(labels),'facts','boundaries'])$(side+'-'+key).value=a[side]?.[key]||''}$('preference').value=a.preference||'';$('comment').value=a.comment||'';loading=false;progress()}
$('cases').onchange=()=>show(Number($('cases').value));$('prev').onclick=()=>show(current-1);$('next').onclick=()=>show(current+1);$('preference').onchange=save;$('comment').oninput=save;
$('download').onclick=()=>{save();const count=pack.pairs.filter(p=>valid(p,answers[p.id])).length,complete=count===30&&$('rater').value.trim()&&$('human').checked;const result={schema:'hmph-human-review-v1',pack_id:pack.id,rater_id:$('rater').value.trim(),human_attestation:$('human').checked,started_at:started,exported_at:new Date().toISOString(),complete:!!complete,valid_pairs:count,trial:{used:$('tried').value==='yes',start:$('trial-start').value,end:$('trial-end').value,notes:$('trial-notes').value},ratings:answers};const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));link.download=`hmph-review-${result.rater_id.replace(/[^a-zA-Z0-9_-]/g,'')||'draft'}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);$('notice').textContent=complete?'本人评分已导出；请交给协调人，不与另一位评审交换答案。':'已导出草稿，尚未满足真人身份与 30 组有效评分要求。'};
$('restore').onchange=async event=>{try{const r=JSON.parse(await event.target.files[0].text());if(r.pack_id!==pack.id||r.schema!=='hmph-human-review-v1')throw Error();if(!confirm('仅恢复你本人填写的草稿，确认继续？'))return;Object.assign(answers,r.ratings);$('rater').value=r.rater_id||'';$('human').checked=!!r.human_attestation;$('tried').value=r.trial?.used?'yes':'no';$('trial-start').value=r.trial?.start||'';$('trial-end').value=r.trial?.end||'';$('trial-notes').value=r.trial?.notes||'';loading=true;show(current);$('notice').textContent='已恢复本人草稿。'}catch{$('notice').textContent='草稿版本或材料编号不匹配，未载入。'}};
loading=true;show(0);
</script></main></html>"""  # noqa: E501


def prepare(source):
    output = source.parent / "human-review.html"
    keyfile = source.parent / "human-review-key.json"
    if output.exists() or keyfile.exists():
        raise ValueError("Review pack already frozen; do not reshuffle or overwrite")
    raw = source.read_bytes()
    report = json.loads(raw)
    assert report["status"] == "completed" and len(report["arms"]) == 2
    arms = {a["arm"]: a["rows"] for a in report["arms"]}
    if any(row.get("reply") == FALLBACK for rows in arms.values() for row in rows):
        raise ValueError("Fix failed conversations before preparing human-review material")
    users = [r for r in arms["off"] if "user" in r]
    mandatory = [
        r for r in users if not r["id"].startswith("after-correction-") or r.get("recurrence")
    ]
    seed = hashlib.sha256(raw).hexdigest()
    optional = sorted(
        [r for r in users if r not in mandatory],
        key=lambda r: hashlib.sha256((seed + r["id"]).encode()).hexdigest(),
    )
    chosen = mandatory + optional[: 30 - len(mandatory)]
    assert len(chosen) == 30
    rng = secrets.SystemRandom()
    rng.shuffle(chosen)
    pairs, mapping = [], {}
    for index, target in enumerate(chosen, 1):
        identifier = f"pair-{index:03}"
        off_first = rng.choice((True, False))
        row = {
            "id": identifier,
            "user": target["user"],
            "family": "C3"
            if target.get("recurrence") or target["id"] in ("name", "correction")
            else "C2"
            if target["id"]
            in (
                "create",
                "return",
                "pause",
                "paused-return",
                "resume",
                "resumed-return",
                "after-forget",
            )
            else "C1",
        }
        mapping[identifier] = {
            "scene_id": target["id"],
            "A": "off" if off_first else "on",
            "B": "on" if off_first else "off",
        }
        for side in ("A", "B"):
            arm = mapping[identifier][side]
            context = []
            for previous in arms[arm]:
                if previous["id"] == target["id"]:
                    if previous.get("new_chat"):
                        context.append("── 当前回复发生在新会话 ──")
                    row[side] = {"reply": previous["reply"], "context": "\n\n".join(context)}
                    row["recorded_at"] = previous["input"].get("reference_time")
                    break
                if previous.get("new_chat"):
                    context.append("── 新会话 ──")
                if "user" in previous:
                    context.extend(["用户：" + previous["user"], "Hmph：" + previous["reply"]])
                else:
                    context.append("界面操作：查看共同计划来源并忘记该计划；回执完成。")
        pairs.append(row)
    pack_id = hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).hexdigest()
    payload = {"id": pack_id, "pairs": pairs}
    encoded = (
        json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    )
    output.write_text(PAGE.replace("__PACK__", encoded), encoding="utf-8")
    keyfile.write_text(
        json.dumps(
            {
                "pack_id": pack_id,
                "source_sha256": seed,
                "source": str(source),
                "pairs": mapping,
                "randomization": "SystemRandom; fixed once; no model grades used for selection",
                "independent_trajectories": 1,
                "mechanism": "C1; C2/C3 behavioral inspection only",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"page": str(output), "pairs": len(pairs), "human_ratings": 0}, ensure_ascii=False
        )
    )


def analyze(source, paths):
    coordinator = json.loads((source.parent / "human-review-key.json").read_text(encoding="utf-8"))
    if coordinator.get("status") == "withdrawn":
        raise ValueError("Review pack withdrawn: fix conversation quality before human review")
    expected = coordinator["pairs"]
    raters = set()
    results = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("schema") != "hmph-human-review-v1"
            or data.get("pack_id") != coordinator["pack_id"]
        ):
            raise ValueError("Different or invalid review pack")
        rater = data.get("rater_id", "").strip()
        if not rater or rater in raters or data.get("human_attestation") is not True:
            raise ValueError("Distinct real reviewers must attest and identify themselves")
        raters.add(rater)
        ratings = data.get("ratings", {})
        if set(ratings) != set(expected) or data.get("complete") is not True:
            raise ValueError("Each reviewer must complete the same 30 pairs")
        votes = Counter()
        scores = {arm: {d: [] for d in DIMENSIONS} for arm in ("off", "on")}
        issues = []
        for identifier, rating in ratings.items():
            choice = rating.get("preference")
            if choice not in ("A", "B", "tie"):
                raise ValueError("Skipped or incomplete preference")
            votes[expected[identifier][choice] if choice != "tie" else "tie"] += 1
            for side in ("A", "B"):
                arm = expected[identifier][side]
                for dimension in DIMENSIONS:
                    value = rating[side][dimension]
                    if dimension == "repair" and value == "na":
                        continue
                    if value not in ("1", "2", "3", "4", "5"):
                        raise ValueError("Invalid rating")
                    scores[arm][dimension].append(int(value))
                if rating[side].get("facts") != "none" or rating[side].get("boundaries") != "none":
                    issues.append(
                        {
                            "pair": identifier,
                            "arm": arm,
                            "facts": rating[side].get("facts"),
                            "boundaries": rating[side].get("boundaries"),
                            "comment": rating.get("comment", ""),
                        }
                    )
        results.append(
            {
                "rater": rater,
                "preferences": dict(votes),
                "mean_scores": {
                    a: {d: sum(v) / len(v) if v else None for d, v in ds.items()}
                    for a, ds in scores.items()
                },
                "issues": issues,
                "trial": data.get("trial"),
            }
        )
    print(
        json.dumps(
            {
                "participation_gate": len(raters) >= 2,
                "raters": len(raters),
                "pairs_per_rater": 30,
                "independent_trajectories": 1,
                "results": results,
                "decision": "assess P5-c from actual human results; no automatic acceptance",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--ratings", type=Path, nargs="+")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.source)
    elif args.ratings:
        analyze(args.source, args.ratings)
    else:
        parser.error("Choose --prepare or --ratings")
